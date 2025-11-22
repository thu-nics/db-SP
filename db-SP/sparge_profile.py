import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
from spas_sage_attn import customize_spas_sage_attn_meansim_cuda

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


# 白色是1
data = torch.load("/mnt/public/chensiqi/query_key_value_dict.pt", map_location="cpu")
query = data['query']  # [1, 75648, 40, 128]
key = data['key']
value = data['value']
device = torch.device("cuda:0")  # 或你想用的 GPU
query = query.to(device)
key = key.to(device)
value = value.to(device)
print(query.shape, key.shape, value.shape)
hidden_states, lut = customize_spas_sage_attn_meansim_cuda(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    smooth_k=True,
    simthreshd1=0.1,
    cdfthreshd=0.9,
    pvthreshd=10000,
    attention_sink=False,
    tensor_layout="HND",
    output_dtype=torch.float16,
    return_sparsity=False,
    return_lut = True
)
print(lut.shape)
print(lut)
print("lut中0的个数:", (lut == 0).sum().item())

import numpy as np
import matplotlib.pyplot as plt

def lut_to_mask(lut):
    # lut形状: [1, 40, 256, 512]
    batch, heads, rows, cols = lut.shape
    masks = torch.zeros(heads, rows, cols)
    
    for h in range(heads):
        for r in range(rows):
            current_pos = 0
            for c in range(cols):
                current_pos += lut[0, h, r, c].item()
                if current_pos < cols:
                    masks[h, r, current_pos] = 1
                else:
                    break
                    
    return masks

mask = lut_to_mask(lut)

fig, axes = plt.subplots(8, 5, figsize=(15, 24))
fig.suptitle('Visualization of 40 Attention Heads LUT Masks', fontsize=16)

output_dir = "lut_visualization"
os.makedirs(output_dir, exist_ok=True)

for i in range(40):
    row, col = i // 5, i % 5
    ax = axes[row, col]
    
    # 显示当前头的掩码
    im = ax.imshow(mask[i].numpy(), cmap='viridis', aspect='auto')
    ax.set_title(f'Head {i}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Row')
    
    # 每行只显示部分标签以避免拥挤
    if col == 0:
        ax.set_ylabel('Row')
    if row == 7:
        ax.set_xlabel('Position')

plt.tight_layout()
plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)

# 保存图像到文件
plt.savefig(os.path.join(output_dir, 'all_heads_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"可视化已保存到 {os.path.join(output_dir, 'all_heads_visualization.png')}")

# 额外：为每个头单独保存图像
print("正在为每个头单独保存图像...")
for i in range(40):
    plt.figure(figsize=(10, 6))
    plt.imshow(mask[i].numpy(), cmap='viridis', aspect='auto')
    plt.title(f'Head {i} LUT Mask Pattern')
    plt.xlabel('Position')
    plt.ylabel('Row')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f'head_{i}_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"所有单独头部的可视化已保存到 {output_dir} 目录")