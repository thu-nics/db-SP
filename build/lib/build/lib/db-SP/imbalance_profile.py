import torch
import matplotlib.pyplot as plt
import math

def imbalance_ratio(sparse:torch.Tensor,num_groups:int=8):
    # for sparge input shape [head, head_density]
    group_sums = sparse.view(sparse.shape[0], num_groups, -1).sum(dim=-1)
    x = group_sums.float().max(dim=1).values / group_sums.float().mean(dim=1)
    return x.float().mean()

def greedy_partition_and_rearrange_multi(sparse: torch.Tensor, num_groups: int = 8, old_perm_idx = None, old_deperm_idx = None):
    """
    输入: sparse [num_blocks, num_heads]
    对每个 block 的 num_heads 按贪心法分组，返回重排后的 sparse、groups、group_sums、perm_idx、deperm_idx
    """
    if sparse.dim() == 4:
        sparse = sparse.sum(dim=(-1,-2))
    num_blocks, B = sparse.shape
    group_size = B // num_groups

    sparse_reordered = []
    all_groups = []
    all_group_sums = []
    all_perm_idx = []
    all_deperm_idx = []

    for block in range(num_blocks):
        weights = sparse[block]
        order = torch.argsort(weights, descending=True)
        w_list = weights[order].detach().cpu().tolist()
        idx_list = order.detach().cpu().tolist()

        groups = [[] for _ in range(num_groups)]
        group_sums = [0.0] * num_groups
        group_counts = [0] * num_groups

        for idx, w in zip(idx_list, w_list):
            gid = min(
                (g for g in range(num_groups) if group_counts[g] < group_size),
                key=lambda g: group_sums[g]
            )
            groups[gid].append(idx)
            group_sums[gid] += float(w)
            group_counts[gid] += 1

        new_order = [i for g in groups for i in g]
        perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)
        deperm_idx = torch.empty_like(perm_idx)
        deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

        sparse_reordered.append(weights.index_select(0, perm_idx))
        all_groups.append(groups)
        all_group_sums.append(group_sums)
        if old_perm_idx is not None and old_deperm_idx is not None:
            all_perm_idx.append(old_perm_idx[block].index_select(0, perm_idx))
            all_deperm_idx.append(deperm_idx.index_select(0, old_deperm_idx[block]))
        else:
            all_perm_idx.append(perm_idx)
            all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    
    return sparse_reordered, all_groups, all_group_sums, all_perm_idx, all_deperm_idx

def imbalance_ratio_hybrid(sparse:torch.Tensor,ulysses_degree:int=2, ring_degree:int=4):
    num_devices = ulysses_degree*ring_degree
    # for paro, input shape [head, height, width]
    if sparse.dim() == 2:
        assert ulysses_degree == 1
        sparse = sparse.unsqueeze(0)
        head, height, width = sparse.shape
    elif sparse.dim() == 3:
        head, height, width = sparse.shape
    elif sparse.dim() == 4:
        batch, head, height, width = sparse.shape
    #pad the last two dimension to be divisible by ring_degree
    if height % ring_degree != 0:
        pad_h = ring_degree - (height % ring_degree)
    else:
        pad_h = 0
    if width % ring_degree != 0:
        pad_w = ring_degree - (width % ring_degree)
    else:
        pad_w = 0
    if pad_h != 0 or pad_w != 0:
        if sparse.dim() == 3:
            sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
        elif sparse.dim() == 4:
            sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
    block_h = height // ring_degree
    block_w = width // ring_degree
    sums = torch.zeros((ulysses_degree, ring_degree, ring_degree), device=sparse.device)
    for k in range(ulysses_degree):    
        for i in range(ring_degree):
            for j in range(ring_degree):
                start_h = i * block_h
                end_h = (i + 1) * block_h
                start_w = j * block_w
                end_w = (j + 1) * block_w
                start_head = k * (head // ulysses_degree)
                end_head = (k + 1) * (head // ulysses_degree)
                if sparse.dim() == 3:
                    sums[k,i,j] += sparse[start_head:end_head, start_h:end_h, start_w:end_w].sum()
                elif sparse.dim() == 4:
                    sums[k,i,j] += sparse[:, start_head:end_head, start_h:end_h, start_w:end_w].sum()
    if ring_degree == 1:
        sums = sums.squeeze(-1).squeeze(-1)
        x = sums.float().max() / sums.float().mean()
        return x
    elif ring_degree == 2:
        iter_1 = torch.maximum(sums[:,0,0], sums[:,1,1])
        iter_2 = torch.maximum(sums[:,0,1], sums[:,1,0])
        max = (iter_1 + iter_2).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x
    elif ring_degree == 4:
        iter_1 = torch.stack([sums[:,0,0], sums[:,1,1], sums[:,2,2], sums[:,3,3]], dim=0).max(dim=0).values
        iter_2 = torch.stack([sums[:,0,1], sums[:,1,2], sums[:,2,3], sums[:,3,0]], dim=0).max(dim=0).values
        iter_3 = torch.stack([sums[:,0,2], sums[:,1,3], sums[:,2,0], sums[:,3,1]], dim=0).max(dim=0).values
        iter_4 = torch.stack([sums[:,0,3], sums[:,1,0], sums[:,2,1], sums[:,3,2]], dim=0).max(dim=0).values
        max = (iter_1 + iter_2 + iter_3 + iter_4).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x
    elif ring_degree == 8:
        iter_1 = torch.stack([sums[:,0,0], sums[:,1,1], sums[:,2,2], sums[:,3,3], sums[:,4,4], sums[:,5,5], sums[:,6,6], sums[:,7,7]], dim=0).max(dim=0).values
        iter_2 = torch.stack([sums[:,0,1], sums[:,1,2], sums[:,2,3], sums[:,3,4], sums[:,4,5], sums[:,5,6], sums[:,6,7], sums[:,7,0]], dim=0).max(dim=0).values
        iter_3 = torch.stack([sums[:,0,2], sums[:,1,3], sums[:,2,4], sums[:,3,5], sums[:,4,6], sums[:,5,7], sums[:,6,0], sums[:,7,1]], dim=0).max(dim=0).values
        iter_4 = torch.stack([sums[:,0,3], sums[:,1,4], sums[:,2,5], sums[:,3,6], sums[:,4,7], sums[:,5,0], sums[:,6,1], sums[:,7,2]], dim=0).max(dim=0).values
        iter_5 = torch.stack([sums[:,0,4], sums[:,1,5], sums[:,2,6], sums[:,3,7], sums[:,4,0], sums[:,5,1], sums[:,6,2], sums[:,7,3]], dim=0).max(dim=0).values
        iter_6 = torch.stack([sums[:,0,5], sums[:,1,6], sums[:,2,7], sums[:,3,0], sums[:,4,1], sums[:,5,2], sums[:,6,3], sums[:,7,4]], dim=0).max(dim=0).values
        iter_7 = torch.stack([sums[:,0,6], sums[:,1,7], sums[:,2,0], sums[:,3,1], sums[:,4,2], sums[:,5,3], sums[:,6,4], sums[:,7,5]], dim=0).max(dim=0).values
        iter_8 = torch.stack([sums[:,0,7], sums[:,1,0], sums[:,2,1], sums[:,3,2], sums[:,4,3], sums[:,5,4], sums[:,6,5], sums[:,7,6]], dim=0).max(dim=0).values
        max = (iter_1 + iter_2 + iter_3 + iter_4 + iter_5 + iter_6 + iter_7 + iter_8).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x

# def greedy_partition_and_rearrange(sparse: torch.Tensor, num_groups: int = 8):
#     """
#     将 [B, H, W] 的 sparse（B=40）按贪心法分到 num_groups 组，并重排为
#     [B, H, W]，使得前 group_size 个属于组0，接着 group_size 个属于组1，依此类推。
#     每组强制恰好 group_size 个元素。
#     返回 deperm_idx，可用于恢复原顺序。
#     """
#     B = sparse.shape[0]
#     group_size = B // num_groups
    
#     weights = sparse.sum(dim=(1, 2))
#     order = torch.argsort(weights, descending=True)
#     w_list = weights[order].detach().cpu().tolist()
#     idx_list = order.detach().cpu().tolist()

#     groups = [[] for _ in range(num_groups)]
#     group_sums = [0.0] * num_groups
#     group_counts = [0] * num_groups

#     for idx, w in zip(idx_list, w_list):
#         gid = min(
#             (g for g in range(num_groups) if group_counts[g] < group_size),
#             key=lambda g: group_sums[g]
#         )
#         groups[gid].append(idx)
#         group_sums[gid] += float(w)
#         group_counts[gid] += 1

#     new_order = [i for g in groups for i in g]
#     perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)

#     # 生成 deperm_idx
#     deperm_idx = torch.empty_like(perm_idx)
#     deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

#     sparse_reordered = sparse.index_select(0, perm_idx)

#     return sparse_reordered, groups, group_sums, perm_idx, deperm_idx

# def ring_permute(sparse: torch.Tensor, num_groups: int = 8):
#     """
#     对 [H, W] 的稀疏 mask，分别对行和列做贪心分组重排，使得每组的总和尽量均衡。
#     返回 permuted_sparse, row_groups, col_groups, row_perm_idx, col_perm_idx, row_deperm_idx, col_deperm_idx
#     """
#     H, W = sparse.shape
#     group_size_h = H // num_groups
#     group_size_w = W // num_groups

#     # 行分组
#     row_sum = sparse.sum(dim=1)
#     row_order = torch.argsort(row_sum, descending=True)
#     row_w_list = row_sum[row_order].detach().cpu().tolist()
#     row_idx_list = row_order.detach().cpu().tolist()
#     row_groups = [[] for _ in range(num_groups)]
#     row_group_sums = [0.0] * num_groups
#     row_group_counts = [0] * num_groups
#     for idx, w in zip(row_idx_list, row_w_list):
#         gid = min(
#             (g for g in range(num_groups) if row_group_counts[g] < group_size_h),
#             key=lambda g: row_group_sums[g]
#         )
#         row_groups[gid].append(idx)
#         row_group_sums[gid] += float(w)
#         row_group_counts[gid] += 1
#     row_new_order = [i for g in row_groups for i in g]
#     row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
#     row_deperm_idx = torch.empty_like(row_perm_idx)
#     row_deperm_idx[row_perm_idx] = torch.arange(len(row_perm_idx), device=row_perm_idx.device)

#     # 列分组
#     col_sum = sparse.sum(dim=0)
#     col_order = torch.argsort(col_sum, descending=True)
#     col_w_list = col_sum[col_order].detach().cpu().tolist()
#     col_idx_list = col_order.detach().cpu().tolist()
#     col_groups = [[] for _ in range(num_groups)]
#     col_group_sums = [0.0] * num_groups
#     col_group_counts = [0] * num_groups
#     for idx, w in zip(col_idx_list, col_w_list):
#         gid = min(
#             (g for g in range(num_groups) if col_group_counts[g] < group_size_w),
#             key=lambda g: col_group_sums[g]
#         )
#         col_groups[gid].append(idx)
#         col_group_sums[gid] += float(w)
#         col_group_counts[gid] += 1
#     col_new_order = [i for g in col_groups for i in g]
#     col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
#     col_deperm_idx = torch.empty_like(col_perm_idx)
#     col_deperm_idx[col_perm_idx] = torch.arange(len(col_perm_idx), device=col_perm_idx.device)

#     # 行列重排
#     permuted_sparse = sparse.index_select(0, row_perm_idx).index_select(1, col_perm_idx)

#     return permuted_sparse, row_groups, col_groups, row_perm_idx, col_perm_idx, row_deperm_idx, col_deperm_idx
    
# def ring_permute_multihead(sparse: torch.Tensor, num_groups: int = 8):
#     """
#     输入: sparse [head, H, W]
#     对每个 head 分别做 ring_permute，返回
#     permuted_sparse [head, H, W]，
#     row_groups_list, col_groups_list, row_perm_idx_list, col_perm_idx_list, row_deperm_idx_list, col_deperm_idx_list
#     """
#     num_heads, H, W = sparse.shape
#     permuted_list = []
#     row_groups_list = []
#     col_groups_list = []
#     row_perm_idx_list = []
#     col_perm_idx_list = []
#     row_deperm_idx_list = []
#     col_deperm_idx_list = []
#     for h in range(num_heads):
#         permuted, row_groups, col_groups, row_perm_idx, col_perm_idx, row_deperm_idx, col_deperm_idx = ring_permute(sparse[h], num_groups)
#         permuted_list.append(permuted)
#         row_groups_list.append(row_groups)
#         col_groups_list.append(col_groups)
#         row_perm_idx_list.append(row_perm_idx)
#         col_perm_idx_list.append(col_perm_idx)
#         row_deperm_idx_list.append(row_deperm_idx)
#         col_deperm_idx_list.append(col_deperm_idx)
#     permuted_sparse = torch.stack(permuted_list, dim=0)
#     return permuted_sparse, row_groups_list, col_groups_list, row_perm_idx_list, col_perm_idx_list, row_deperm_idx_list, col_deperm_idx_list

def hybrid_permute(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4
):
    num_heads, H, W = sparse.shape
    # 1. head 维度贪心分组重排
    if ulysses_degree == 1:
        head_perm_idx = torch.arange(num_heads, device=sparse.device)
        head_deperm_idx = torch.arange(num_heads, device=sparse.device)
        sparse_reordered = sparse
    else:
        head_group_size = num_heads // ulysses_degree
        head_weights = sparse.sum(dim=(1,2))
        head_order = torch.argsort(head_weights, descending=True)
        head_w_list = head_weights[head_order].detach().cpu().tolist()
        head_idx_list = head_order.detach().cpu().tolist()
        head_groups = [[] for _ in range(ulysses_degree)]
        head_group_sums = [0.0] * ulysses_degree
        head_group_counts = [0] * ulysses_degree
        for idx, w in zip(head_idx_list, head_w_list):
            gid = min(
                (g for g in range(ulysses_degree) if head_group_counts[g] < head_group_size),
                key=lambda g: head_group_sums[g]
            )
            head_groups[gid].append(idx)
            head_group_sums[gid] += float(w)
            head_group_counts[gid] += 1
        head_new_order = [i for g in head_groups for i in g]
        head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
        head_deperm_idx = torch.empty_like(head_perm_idx)
        head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
        sparse_reordered = sparse.index_select(0, head_perm_idx)

    # 2. 对每个head做H/W贪心分组重排
    if ring_degree == 1:
        row_perm_idx_list = [torch.arange(H, device=sparse.device) for _ in range(num_heads)]
        col_perm_idx_list = [torch.arange(W, device=sparse.device) for _ in range(num_heads)]
        row_deperm_idx_list = [torch.arange(H, device=sparse.device) for _ in range(num_heads)]
        col_deperm_idx_list = [torch.arange(W, device=sparse.device) for _ in range(num_heads)]
        sparse_final = sparse_reordered
    else:
        assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"
        row_perm_idx_list = []
        col_perm_idx_list = []
        row_deperm_idx_list = []
        col_deperm_idx_list = []
        permuted_list = []
        for h in range(num_heads):
            mat = sparse_reordered[h]
            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)
            row_order = torch.argsort(row_sum, descending=True)
            row_w_list = row_sum[row_order].detach().cpu().tolist()
            row_idx_list = row_order.detach().cpu().tolist()
            row_groups = [[] for _ in range(ring_degree)]
            row_group_sums = [0.0] * ring_degree
            row_group_counts = [0] * ring_degree
            for idx, w in zip(row_idx_list, row_w_list):
                gid = min(
                    (g for g in range(ring_degree) if row_group_counts[g] < group_size_h),
                    key=lambda g: row_group_sums[g]
                )
                row_groups[gid].append(idx)
                row_group_sums[gid] += float(w)
                row_group_counts[gid] += 1
            row_new_order = [i for g in row_groups for i in g]
            row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
            row_deperm_idx = torch.empty_like(row_perm_idx)
            row_deperm_idx[row_perm_idx] = torch.arange(len(row_perm_idx), device=row_perm_idx.device)
            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)
            col_order = torch.argsort(col_sum, descending=True)
            col_w_list = col_sum[col_order].detach().cpu().tolist()
            col_idx_list = col_order.detach().cpu().tolist()
            col_groups = [[] for _ in range(ring_degree)]
            col_group_sums = [0.0] * ring_degree
            col_group_counts = [0] * ring_degree
            for idx, w in zip(col_idx_list, col_w_list):
                gid = min(
                    (g for g in range(ring_degree) if col_group_counts[g] < group_size_w),
                    key=lambda g: col_group_sums[g]
                )
                col_groups[gid].append(idx)
                col_group_sums[gid] += float(w)
                col_group_counts[gid] += 1
            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
            col_deperm_idx = torch.empty_like(col_perm_idx)
            col_deperm_idx[col_perm_idx] = torch.arange(len(col_perm_idx), device=col_perm_idx.device)
            mat_permuted = mat.index_select(0, row_perm_idx).index_select(1, col_perm_idx)
            permuted_list.append(mat_permuted)
            row_perm_idx_list.append(row_perm_idx)
            col_perm_idx_list.append(col_perm_idx)
            row_deperm_idx_list.append(row_deperm_idx)
            col_deperm_idx_list.append(col_deperm_idx)
        sparse_final = torch.stack(permuted_list, dim=0)
    return sparse_final, head_perm_idx, row_perm_idx_list, col_perm_idx_list, head_deperm_idx, row_deperm_idx_list, col_deperm_idx_list 

device = "cuda" if torch.cuda.is_available() else "cpu"
sparse_data = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True)
sparse = sparse_data['sparse'][0, :, :, :, :].to(device)  # [40, 40, 1182, 1182]

u8r1_list = []
u4r2_list = []
u2r4_list = []
u1r8_list = []
u4r1_list = []
u2r2_list = []
u4r2_reorder_list = []

_, all_groups, all_group_sums, all_perm_idx, all_deperm_idx = greedy_partition_and_rearrange_multi(sparse,num_groups=4)
print(sparse.shape)
sparse_reordered = torch.stack(
    [sparse[block].index_select(0, all_perm_idx[block]) for block in range(sparse.shape[0])]
)
print(sparse_reordered.shape)

print("-----ulysses8ring1-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=8, ring_degree=1)
    u8r1_list.append(x)
    # print(f"block {block}: {x}")
print("-----ulysses4ring2-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=4, ring_degree=2)
    u4r2_list.append(x)
    # print(f"block {block}: {x}")
print("-----ulysses2ring4-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=2, ring_degree=4)
    u2r4_list.append(x)
    # print(f"block {block}: {x}")
print("-----ulysses1ring8-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=1, ring_degree=8)
    u1r8_list.append(x)
    # print(f"block {block}: {x}")
print("-----ulysses4ring1-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=4, ring_degree=1)
    u4r1_list.append(x)
    #print(f"block {block}: {x}")
print("-----ulysses2ring2-----")
for block in range(sparse.shape[0]):
    sparse_block = sparse[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=2, ring_degree=2)
    u2r2_list.append(x)
    #print(f"block {block}: {x}")
print("-----ulysses4ring2_reorder-----")
for block in range(sparse_reordered.shape[0]):
    sparse_block = sparse_reordered[block]
    x=imbalance_ratio_hybrid(sparse_block, ulysses_degree=4, ring_degree=2)
    u4r2_reorder_list.append(x)
    # print(f"block {block}: {x}")

print(f"average imbalance ratio ulysses8ring1: {torch.tensor(u8r1_list).mean()}")
print(f"average imbalance ratio ulysses4ring2: {torch.tensor(u4r2_list).mean()}")
print(f"average imbalance ratio ulysses2ring4: {torch.tensor(u2r4_list).mean()}")
print(f"average imbalance ratio ulysses1ring8: {torch.tensor(u1r8_list).mean()}")
print(f"average imbalance ratio ulysses4ring1: {torch.tensor(u4r1_list).mean()}")
print(f"average imbalance ratio ulysses2ring2: {torch.tensor(u2r2_list).mean()}")
print(f"average imbalance ratio ulysses4ring2_reorder: {torch.tensor(u4r2_reorder_list).mean()}")
