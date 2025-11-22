import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
#from spas_sage_attn import customize_spas_sage_attn_meansim_cuda

def rearrange_tensor_optimized(q: torch.Tensor, num_groups: int = 8):
    """
    高效重排张量，第[:,:,0:5,:]保持不变，其他分组在第一个维度上进行循环移位
    
    参数:
        q: 形状为 [1, 75648, 40, 128] 的张量
        
    返回:
        重排后的张量
    """
    # 获取张量形状信息
    batch_size, seq_len, channels, feature_dim = q.shape
    
    # 每5个通道为一组，共8组
    channels_per_group = channels // num_groups
    seq_len_per_group = seq_len // num_groups
    
    # 每组需要移动的位置数
    shift_amounts = [0] + [seq_len_per_group * i for i in range(1, num_groups)]
    
    # 预分配结果张量
    result = torch.empty_like(q)
    
    # 处理每个分组
    for group_idx in range(num_groups):
        start_channel = group_idx * channels_per_group
        end_channel = (group_idx + 1) * channels_per_group
        
        if group_idx == 0:
            # 第0组保持不变
            result[:, :, start_channel:end_channel, :] = q[:, :, start_channel:end_channel, :]
        else:
            # 使用roll操作实现高效的循环移位
            shift = shift_amounts[group_idx]
            result[:, :, start_channel:end_channel, :] = torch.roll(
                q[:, :, start_channel:end_channel, :], 
                shifts=-shift, 
                dims=1
            )
    
    return result

def greedy_partition_and_rearrange_ulysses8(sparse: torch.Tensor, num_groups: int = 8, group_size: int = 5):
    """
    将 [B, H, W] 的 sparse（B=40）按贪心法分到 num_groups 组，并重排为
    [B, H, W]，使得前 group_size 个属于组0，接着 group_size 个属于组1，依此类推。
    每组强制恰好 group_size 个元素。
    返回 deperm_idx，可用于恢复原顺序。
    """
    B = sparse.shape[0]
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

    weights = sparse.sum(dim=(1, 2))
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

    # 生成 deperm_idx
    deperm_idx = torch.empty_like(perm_idx)
    deperm_idx[perm_idx] = torch.arange(len(perm_idx), device=perm_idx.device)

    sparse_reordered = sparse.index_select(0, perm_idx)

    return sparse_reordered, groups, group_sums, perm_idx, deperm_idx

def greedy_partition_and_rearrange_ulysses8_multi(sparse: torch.Tensor, num_groups: int = 8, group_size: int = 5):
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
        all_perm_idx.append(perm_idx)
        all_deperm_idx.append(deperm_idx)

    sparse_reordered = torch.stack(sparse_reordered, dim=0)
    return sparse_reordered, all_groups, all_group_sums, all_perm_idx, all_deperm_idx

def x_ulysses8ring1 (sparse:torch.Tensor):
    sums = sparse.sum(dim=(1,2))
    group_sums = sums.view(8, 5).sum(dim=1)
    x = group_sums.float().max() / group_sums.float().mean()
    return x

# 使用示例
# q = torch.randn(1, 1184, 40, 1184)  # 您的输入张量
# for i in range(10):  # 预热
#     _ = rearrange_tensor_optimized(q)
# start = time.perf_counter()
# result = rearrange_tensor_optimized(q)
# elapsed = time.perf_counter() - start
# print(f"latency: {elapsed*1e3:.6f} ms")

# 白色是1
import torch
import matplotlib.pyplot as plt

# 加载两个 list
list2 = torch.load('/mnt/public/chensiqi/ParaAttention/results/parallel/total_time_list_2.pt')
list4 = torch.load('/mnt/public/chensiqi/ParaAttention/results/parallel/total_time_list_4.pt')
list8 = torch.load('/mnt/public/chensiqi/ParaAttention/results/parallel/total_time_list_8.pt')
list8_original = torch.load('/mnt/public/chensiqi/ParaAttention/results/parallel/total_time_list_8_original.pt')
list2_norm = [x/4 for x in list2]  
list4_norm = [x/2 for x in list4]  
list8_norm = [x/1 for x in list8]  

plt.figure()
plt.plot(list2_norm, marker='o', label='total_time_list_2')
plt.plot(list4_norm, marker='s', label='total_time_list_4')
plt.plot(list8_norm, marker='^', label='total_time_list_8')
plt.xlabel('Timestep')
plt.ylabel('Total Attention Time (ms)')
plt.title('Total Attention Time per Timestep')
plt.grid(True)
plt.legend()
plt.savefig('/mnt/public/chensiqi/ParaAttention/results/parallel/timestep_compare_normalize.png')
plt.close()

plt.figure()
plt.plot(list2, marker='o', label='total_time_list_2')
plt.plot(list4, marker='s', label='total_time_list_4')
plt.plot(list8, marker='^', label='total_time_list_8')
plt.xlabel('Timestep')
plt.ylabel('Total Attention Time (ms)')
plt.title('Total Attention Time per Timestep')
plt.grid(True)
plt.legend()
plt.savefig('/mnt/public/chensiqi/ParaAttention/results/parallel/timestep_compare.png')
plt.close()

plt.figure()
plt.plot(list8, marker='o', label='total_time_list_8')
plt.plot(list8_original, marker='s', label='total_time_list_8_original')
plt.xlabel('Timestep')
plt.ylabel('Total Attention Time (ms)')
plt.title('Total Attention Time per Timestep')
plt.grid(True)
plt.legend()
plt.savefig('/mnt/public/chensiqi/ParaAttention/results/parallel/timestep_compare2.png')
plt.close()

original_time = sum(list8_original)
optimized_time = sum(list8)
print(f"Original total time: {original_time:.2f} ms")
print(f"Optimized total time: {optimized_time:.2f} ms")
print(f"Speedup: {original_time/optimized_time:.2f}x")