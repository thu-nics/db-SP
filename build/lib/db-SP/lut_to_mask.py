import torch
import math
import matplotlib.pyplot as plt

def lut_to_mask(lut):
    batch, heads, rows, cols = lut.shape
    masks = torch.zeros(batch, heads, rows, cols, device=lut.device, dtype=torch.bool)
    
    # 逐批次处理，减少内存使用
    for b in range(batch):
        batch_lut = lut[b]  # [heads, rows, cols]
        batch_cumulative = torch.cumsum(batch_lut, dim=-1)  # [heads, rows, cols]
        
        # 只处理有效位置
        valid_mask = batch_cumulative < cols
        
        # 获取有效位置的索引
        valid_indices = batch_cumulative[valid_mask].long()
        
        # 创建对应的行和头索引
        head_indices, row_indices = torch.meshgrid(
            torch.arange(heads, device=lut.device),
            torch.arange(rows, device=lut.device),
            indexing='ij'
        )
        head_indices = head_indices.flatten().unsqueeze(-1).expand(-1, cols).flatten()
        row_indices = row_indices.flatten().unsqueeze(-1).expand(-1, cols).flatten()
        col_indices = torch.arange(cols, device=lut.device).repeat(heads * rows)
        
        # 创建完整索引
        full_indices = torch.stack([
            head_indices[valid_mask.flatten()],
            row_indices[valid_mask.flatten()],
            valid_indices
        ])
        
        # 设置mask
        masks[b, full_indices[0], full_indices[1], full_indices[2]] = True
    
    return masks

def hybrid_imbalance_ratio(sparse:torch.Tensor,ulysses_degree:int=2, ring_degree:int=4):
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
        return x.item()
    elif ring_degree == 2:
        iter_1 = torch.maximum(sums[:,0,0], sums[:,1,1])
        iter_2 = torch.maximum(sums[:,0,1], sums[:,1,0])
        max = (iter_1 + iter_2).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x.item()
    elif ring_degree == 4:
        iter_1 = torch.stack([sums[:,0,0], sums[:,1,1], sums[:,2,2], sums[:,3,3]], dim=0).max(dim=0).values
        iter_2 = torch.stack([sums[:,0,1], sums[:,1,2], sums[:,2,3], sums[:,3,0]], dim=0).max(dim=0).values
        iter_3 = torch.stack([sums[:,0,2], sums[:,1,3], sums[:,2,0], sums[:,3,1]], dim=0).max(dim=0).values
        iter_4 = torch.stack([sums[:,0,3], sums[:,1,0], sums[:,2,1], sums[:,3,2]], dim=0).max(dim=0).values
        max = (iter_1 + iter_2 + iter_3 + iter_4).max()
        mean = sums.sum() / num_devices
        x = max.float() / mean.float()
        return x.item()
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
        return x.item()

def hybrid_permute_v4(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4,
    reward: float = 2
):
    if sparse.dim() == 3:
        num_heads, H, W = sparse.shape
        # 1. head 维度贪心分组重排
        if ulysses_degree == 1:
            head_perm_idx = None
            head_deperm_idx = None
            sparse_reordered = sparse
            head_group_size = num_heads
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

            # 对每个组内的 heads 按权重从小到大排序
            for g in range(ulysses_degree):
                head_groups[g] = sorted(head_groups[g], key=lambda idx: head_weights[idx].item())

            head_new_order = [i for g in head_groups for i in g]
            head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
            head_deperm_idx = torch.empty_like(head_perm_idx)
            head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
            sparse_reordered = sparse.index_select(0, head_perm_idx)
            # sparse_reordered = sparse

        # 2. 将每组ulysses的head累加为一个head

        mat = sparse_reordered.sum(dim=0) # [H, W]

        # 3. 对每个组累加后的mask做H/W贪心分组重排
        if ring_degree == 1:
            new_row_perm_idx = None
            new_col_perm_idx = None
            transpose_matrix_q = None
            transpose_matrix_k = None
            new_row_deperm_idx = None
            new_col_deperm_idx = None
            sparse_final = sparse_reordered
        else:
            assert H % ring_degree == 0 and W % ring_degree == 0, "H和W必须能被ring_degree整除"

            group_size_h = H // ring_degree
            row_sum = mat.sum(dim=1)
            row_order = torch.argsort(row_sum, descending=True)
            row_w_list = row_sum[row_order].detach().cpu().tolist()
            row_idx_list = row_order.detach().cpu().tolist()
            row_groups = [[] for _ in range(ring_degree)]
            row_group_sums = [0.0] * ring_degree
            row_group_counts = [0] * ring_degree

            for idx, w in zip(row_idx_list, row_w_list):
                # 计算该行原本属于哪个块
                original_block = idx // group_size_h

                # 优先考虑原本的块，如果该块还有空间且负载相对均衡
                candidate_groups = []
                for g in range(ring_degree):
                    if row_group_counts[g] < group_size_h:
                        # 如果是原本的块，给予优先级（负载稍高也可以接受）
                        if g == original_block:
                            candidate_groups.append((g, row_group_sums[g] - reward * w))  # 降低原本块的负载计算
                        else:
                            candidate_groups.append((g, row_group_sums[g]))

                if candidate_groups:
                    gid = min(candidate_groups, key=lambda x: x[1])[0]
                    row_groups[gid].append(idx)
                    row_group_sums[gid] += float(w)
                    row_group_counts[gid] += 1

            row_new_order = [i for g in row_groups for i in g]
            row_perm_idx = torch.tensor(row_new_order, device=sparse.device, dtype=torch.long)
            group_size_w = W // ring_degree
            col_sum = mat.sum(dim=0)
            col_order = torch.argsort(col_sum, descending=True)
            col_w_list = col_sum[col_order].detach().cpu().tolist()
            col_idx_list = col_order.detach().cpu().tolist()
            col_groups = [[] for _ in range(ring_degree)]
            col_group_sums = [0.0] * ring_degree
            col_group_counts = [0] * ring_degree

            for idx, w in zip(col_idx_list, col_w_list):
                original_block = idx // group_size_w
                
                candidate_groups = []
                for g in range(ring_degree):
                    if col_group_counts[g] < group_size_w:
                        if g == original_block:
                            candidate_groups.append((g, col_group_sums[g] -  reward * w))  # 降低原本块的负载计算
                        else:
                            candidate_groups.append((g, col_group_sums[g]))
                
                if candidate_groups:
                    gid = min(candidate_groups, key=lambda x: x[1])[0]
                    col_groups[gid].append(idx)
                    col_group_sums[gid] += float(w)
                    col_group_counts[gid] += 1

            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)

            num_groups = ring_degree
            # group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size_h), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size_w), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size_h) & (g < (j + 1) * group_size_h)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size_w) & (g < (j + 1) * group_size_w)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size_h) & (row_perm_idx_groups_sorted < (i + 1) * group_size_h)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(1, new_row_perm_idx.view(-1))
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size_w) & (col_perm_idx_groups_sorted < (i + 1) * group_size_w)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(2, new_col_perm_idx.view(-1))
            new_col_perm_idx = new_col_perm_idx - new_col_perm_idx.min(dim=1, keepdim=True)[0]

            new_row_deperm_idx = torch.empty_like(new_row_perm_idx)
            for i in range(new_row_perm_idx.shape[0]):
                new_row_deperm_idx[i][new_row_perm_idx[i]] = torch.arange(new_row_perm_idx.shape[1], device=new_row_perm_idx.device)

            new_col_deperm_idx = torch.empty_like(new_col_perm_idx)
            for i in range(new_col_perm_idx.shape[0]):
                new_col_deperm_idx[i][new_col_perm_idx[i]] = torch.arange(new_col_perm_idx.shape[1], device=new_col_perm_idx.device)

            sparse_final = sparse_reordered.index_select(1, row_perm_idx_groups_sorted.view(-1)).index_select(2, col_perm_idx_groups_sorted.view(-1)).contiguous()
            # sparse_final = sparse_reordered
        
        return sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx

    elif sparse.dim() == 4:
        sparse_final_list = []
        head_perm_idx_list = []
        new_row_perm_idx_list = []
        new_col_perm_idx_list = []
        transpose_matrix_q_list = []
        transpose_matrix_k_list = []
        head_deperm_idx_list = []
        new_row_deperm_idx_list = []
        new_col_deperm_idx_list = []

        for block in range(sparse.shape[0]):
            sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v4(sparse[block], ulysses_degree, ring_degree,reward)
            sparse_final_list.append(sparse_final)
            head_perm_idx_list.append(head_perm_idx)
            new_row_perm_idx_list.append(new_row_perm_idx)
            new_col_perm_idx_list.append(new_col_perm_idx)
            transpose_matrix_q_list.append(transpose_matrix_q)
            transpose_matrix_k_list.append(transpose_matrix_k)
            head_deperm_idx_list.append(head_deperm_idx)
            new_row_deperm_idx_list.append(new_row_deperm_idx)
            new_col_deperm_idx_list.append(new_col_deperm_idx)

        sparse_final = torch.stack(sparse_final_list, dim=0).contiguous()
        # head_perm_idx = torch.stack(head_perm_idx_list, dim=0)
        # new_row_perm_idx = torch.stack(new_row_perm_idx_list, dim=0)
        # new_col_perm_idx = torch.stack(new_col_perm_idx_list, dim=0)
        # transpose_matrix_q = torch.stack(transpose_matrix_q_list, dim=0)
        # transpose_matrix_k = torch.stack(transpose_matrix_k_list, dim=0)
        # head_deperm_idx = torch.stack(head_deperm_idx_list, dim=0)
        # new_row_deperm_idx = torch.stack(new_row_deperm_idx_list, dim=0)
        # new_col_deperm_idx = torch.stack(new_col_deperm_idx_list, dim=0)

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list

# lut = torch.load("/root/chensiqi/cogvideo_lut.pt").cuda()
# print(lut)
# print(lut.shape)  # torch.Size([42, 48, 684, 1343])
# total_elements = lut.numel()
# non_zero_elements = torch.count_nonzero(lut)
# ratio = non_zero_elements / total_elements
# print(ratio)
# mask = lut_to_mask(lut)
# print(mask.shape)  # torch.Size([42, 48, 684, 1343])
# print(mask.float().sum()/mask.numel())  
# torch.save(mask, "/root/chensiqi/cogvideo_mask.pt")

# sparse = torch.load("/root/chensiqi/cogvideo_mask.pt").cuda()
# sparse = torch.repeat_interleave(sparse, 2, dim=2)
# torch.save(sparse, "/root/chensiqi/cogvideo_mask.pt")
sparse = torch.load("/root/chensiqi/cogvideo_mask.pt").cuda()
print(sparse.shape)
imbalance_ratio_u8r1 = []
imbalance_ratio_u4r2 = []
imbalance_ratio_u2r4 = []
imbalance_ratio_u1r8 = []
imbalance_ratio_u8r1_new = []
imbalance_ratio_u4r2_new = []
imbalance_ratio_u2r4_new = []
imbalance_ratio_u1r8_new = []
H, W = sparse.shape[-2], sparse.shape[-1]
pad_h = (8 - H % 8) if H % 8 != 0 else 0
pad_w = (8 - W % 8) if W % 8 != 0 else 0
if pad_h != 0 or pad_w != 0:
    sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)

for block in range(sparse.shape[0]):
    sparse_piece = sparse[block]
    for (ulysses_degree, ring_degree) in [(8,1),(4,2),(2,4),(1,8)]:
        sparse_new, _, _, _, _, _, _, _, _ = hybrid_permute_v4(sparse_piece,ulysses_degree,ring_degree,2)
        print(f"{ulysses_degree},{ring_degree},{hybrid_imbalance_ratio(sparse_piece,ulysses_degree,ring_degree)},{hybrid_imbalance_ratio(sparse_new,ulysses_degree,ring_degree)}")
        if ring_degree ==8:
            imbalance_ratio_u1r8_new.append(hybrid_imbalance_ratio(sparse_new,ulysses_degree,ring_degree))
        elif ring_degree ==4:
            imbalance_ratio_u2r4_new.append(hybrid_imbalance_ratio(sparse_new,ulysses_degree,ring_degree))
        elif ring_degree ==2:
            imbalance_ratio_u4r2_new.append(hybrid_imbalance_ratio(sparse_new,ulysses_degree,ring_degree))
        elif ring_degree ==1:
            imbalance_ratio_u8r1_new.append(hybrid_imbalance_ratio(sparse_new,ulysses_degree,ring_degree))
        
    imbalance_ratio_u8r1.append(hybrid_imbalance_ratio(sparse_piece,8,1))
    imbalance_ratio_u4r2.append(hybrid_imbalance_ratio(sparse_piece,4,2))
    imbalance_ratio_u2r4.append(hybrid_imbalance_ratio(sparse_piece,2,4))
    imbalance_ratio_u1r8.append(hybrid_imbalance_ratio(sparse_piece,1,8))
    
print("u8r1:", sum(imbalance_ratio_u8r1)/len(imbalance_ratio_u8r1))
print("u4r2:", sum(imbalance_ratio_u4r2)/len(imbalance_ratio_u4r2))
print("u2r4:", sum(imbalance_ratio_u2r4)/len(imbalance_ratio_u2r4))
print("u1r8:", sum(imbalance_ratio_u1r8)/len(imbalance_ratio_u1r8))
print("u8r1_new:", sum(imbalance_ratio_u8r1_new)/len(imbalance_ratio_u8r1_new))
print("u4r2_new:", sum(imbalance_ratio_u4r2_new)/len(imbalance_ratio_u4r2_new))
print("u2r4_new:", sum(imbalance_ratio_u2r4_new)/len(imbalance_ratio_u2r4_new))
print("u1r8_new:", sum(imbalance_ratio_u1r8_new)/len(imbalance_ratio_u1r8_new))

# # visualization
# block = 0
# sparse_piece = sparse[block]
# num_heads = sparse_piece.shape[0]
# ncols = 5
# nrows = math.ceil(num_heads / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), squeeze=False)

# num_heads = sparse_piece.shape[0]
# ncols = 5
# nrows = math.ceil(num_heads / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), squeeze=False)

# for i in range(num_heads):
#     row, col = divmod(i, ncols)
#     ax = axes[row][col]
#     ax.imshow(sparse_piece[i].cpu().numpy(), cmap='gray', aspect='auto')
#     ax.set_title(f'head {i}')
#     ax.axis('off')

# # 去除多余子图
# for j in range(num_heads, nrows * ncols):
#     row, col = divmod(j, ncols)
#     axes[row][col].axis('off')

# plt.tight_layout()
# plt.savefig(f'/root/chensiqi/cogvideo_{block}.png')
