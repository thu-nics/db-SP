import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
def greedy_partition_and_rearrange_ulysses8_multi(sparse: torch.Tensor, old_perm_idx: list, old_deperm_idx:list, num_groups: int = 8, group_size: int = 5):
    """
    输入: sparse [num_blocks, num_heads]
    对每个 block 的 num_heads 按贪心法分组，返回重排后的 sparse、groups、group_sums、perm_idx、deperm_idx
    """
    num_blocks, B = sparse.shape
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

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
            all_perm_idx.append(old_perm_idx[block].to(perm_idx.device).index_select(0, perm_idx))
            all_deperm_idx.append(deperm_idx.index_select(0, old_deperm_idx[block].to(deperm_idx.device)))
        else:
            all_perm_idx.append(perm_idx)
            all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    
    return sparse_reordered, all_groups, all_group_sums, all_perm_idx, all_deperm_idx

# overhead profile
block = 2
# sparse_data = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True)
# sparse = sparse_data['sparse'][0, block, :, :, :].to("cuda")  
sparse_reordered = torch.load("/mnt/public/chensiqi/sparse_reordered_all.pt")[block, :, :, :].to("cuda")  
perm_idx_all = torch.load("/mnt/public/chensiqi/perm_idx_all.pt")
deperm_idx_all = torch.load("/mnt/public/chensiqi/deperm_idx_all.pt")
perm_idx = perm_idx_all[block].to("cuda")
deperm_idx = deperm_idx_all[block].to("cuda")

head_density = torch.randn(40, 40).to("cuda")
# perm_idx_list = [perm_idx_all[i] for i in range(perm_idx_all.shape[0])]
# deperm_idx_list = [deperm_idx_all[i] for i in range(deperm_idx_all.shape[0])]
perm_idx_list=None
deperm_idx_list=None

batch_size =1
seqlen=18900
nheads=40
d=128
device="cuda"
dtype=torch.float16
q = torch.randn(
    batch_size,
    seqlen,
    nheads,
    d,
    device=device,
    dtype=dtype,
    requires_grad= False,
)
k = torch.randn(
    batch_size,
    seqlen,
    nheads,
    d,
    device=device,
    dtype=dtype,
    requires_grad= False,
)
v = torch.randn(
    batch_size,
    seqlen,
    nheads,
    d,
    device=device,
    dtype=dtype,
    requires_grad= False,
)
o_reordered = torch.randn(
    batch_size,
    seqlen,
    nheads,
    d,
    device=device,
    dtype=dtype,
    requires_grad= False,
)
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        torch.cuda.synchronize()
        for i in range(3):
            # sparse_reordered = sparse.index_select(0, perm_idx)           
            q_reordered=q.index_select(2, perm_idx)
            k_reordered=k.index_select(2, perm_idx)
            v_reordered=v.index_select(2, perm_idx)
            o=o_reordered.index_select(2, deperm_idx)
            torch.cuda.synchronize()
        with record_function("reorder_overhead"):
            q_reordered=q.index_select(2, perm_idx)
            k_reordered=k.index_select(2, perm_idx)
            v_reordered=v.index_select(2, perm_idx)
            o=o_reordered.index_select(2, deperm_idx)
            torch.cuda.synchronize()
        for i in range(3):
            _ , _,_,perm_idx_list, deperm_idx_list=greedy_partition_and_rearrange_ulysses8_multi(head_density, old_perm_idx=perm_idx_list if  perm_idx_list is not None else None,old_deperm_idx=deperm_idx_list if deperm_idx_list is not None else None,group_size=5)
        with record_function("decision_overhead"):
             _ , _,_,perm_idx_list, deperm_idx_list=greedy_partition_and_rearrange_ulysses8_multi(head_density, old_perm_idx=perm_idx_list if  perm_idx_list is not None else None,old_deperm_idx=deperm_idx_list if deperm_idx_list is not None else None,group_size=5)

print(q_reordered.shape)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace(f"overhead_profile/overhead_cogvideo_85k.json")
