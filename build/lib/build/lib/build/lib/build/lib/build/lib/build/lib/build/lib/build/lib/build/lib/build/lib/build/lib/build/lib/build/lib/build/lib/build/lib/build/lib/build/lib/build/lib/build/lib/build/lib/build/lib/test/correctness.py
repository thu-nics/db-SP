import os
from yunchang import LongContextAttention, set_seq_parallel_pg, EXTRACT_FUNC_DICT
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

# try:
#     from flash_attn import flash_attn_func
# except ImportError:
#     raise RuntimeError("flash_attn is necessary for this test!")
from yunchang.kernels import AttnType
from test_utils import attention_ref
import argparse
import time
import matplotlib.pyplot as plt

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

def hybrid_permute_v2(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4
):
    """
    先对head做贪心分组重排，然后将每组ulysses的head累加为一个head（降维），
    再对每个组累加后的mask做H/W贪心分组重排。
    返回 [ulysses_degree, H, W]，以及各维度的perm/deperm idx。
    """
    num_heads, H, W = sparse.shape
    # 1. head 维度贪心分组重排
    if ulysses_degree == 1:
        head_perm_idx = torch.arange(num_heads, device=sparse.device)
        head_deperm_idx = torch.arange(num_heads, device=sparse.device)
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
        head_new_order = [i for g in head_groups for i in g]
        head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
        head_deperm_idx = torch.empty_like(head_perm_idx)
        head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
        sparse_reordered = sparse.index_select(0, head_perm_idx)

    # 2. 将每组ulysses的head累加为一个head

    mat = sparse_reordered.sum(dim=0) # [ulysses_degree, H, W]

    # 3. 对每个组累加后的mask做H/W贪心分组重排
    if ring_degree == 1:
        row_perm_idx = torch.arange(H, device=sparse.device) 
        col_perm_idx = torch.arange(W, device=sparse.device)
        row_deperm_idx = torch.arange(H, device=sparse.device)
        col_deperm_idx = torch.arange(W, device=sparse.device)
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
        sparse_final = sparse_reordered.index_select(1, row_perm_idx).index_select(2, col_perm_idx)

    return sparse_final, head_perm_idx, row_perm_idx, col_perm_idx, head_deperm_idx, row_deperm_idx, col_deperm_idx

def hybrid_permute_v3(
    sparse: torch.Tensor,
    ulysses_degree: int = 2,
    ring_degree: int = 4
):
    if sparse.dim() == 3:
        num_heads, H, W = sparse.shape
        # 1. head 维度贪心分组重排
        if ulysses_degree == 1:
            head_perm_idx = torch.arange(num_heads, device=sparse.device)
            head_deperm_idx = torch.arange(num_heads, device=sparse.device)
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
            head_new_order = [i for g in head_groups for i in g]
            head_perm_idx = torch.tensor(head_new_order, device=sparse.device, dtype=torch.long)
            head_deperm_idx = torch.empty_like(head_perm_idx)
            head_deperm_idx[head_perm_idx] = torch.arange(len(head_perm_idx), device=head_perm_idx.device)
            sparse_reordered = sparse.index_select(0, head_perm_idx)

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
                gid = min(
                    (g for g in range(ring_degree) if row_group_counts[g] < group_size_h),
                    key=lambda g: row_group_sums[g]
                )
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
                gid = min(
                    (g for g in range(ring_degree) if col_group_counts[g] < group_size_w),
                    key=lambda g: col_group_sums[g]
                )
                col_groups[gid].append(idx)
                col_group_sums[gid] += float(w)
                col_group_counts[gid] += 1
            col_new_order = [i for g in col_groups for i in g]
            col_perm_idx = torch.tensor(col_new_order, device=sparse.device, dtype=torch.long)
            # sparse_final = sparse_reordered.index_select(1, row_perm_idx).index_select(2, col_perm_idx)

            num_groups = ring_degree
            group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size) & (row_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(1, new_row_perm_idx.view(-1))
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size) & (col_perm_idx_groups_sorted < (i + 1) * group_size)]
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

            sparse_final = sparse_reordered.index_select(1, row_perm_idx_groups_sorted.view(-1)).index_select(2, col_perm_idx_groups_sorted.view(-1)) 
            #TODO: fix
        
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
            sparse_final, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v3(sparse[block], ulysses_degree, ring_degree)
            sparse_final_list.append(sparse_final)
            head_perm_idx_list.append(head_perm_idx)
            new_row_perm_idx_list.append(new_row_perm_idx)
            new_col_perm_idx_list.append(new_col_perm_idx)
            transpose_matrix_q_list.append(transpose_matrix_q)
            transpose_matrix_k_list.append(transpose_matrix_k)
            head_deperm_idx_list.append(head_deperm_idx)
            new_row_deperm_idx_list.append(new_row_deperm_idx)
            new_col_deperm_idx_list.append(new_col_deperm_idx)

        sparse_final = torch.stack(sparse_final_list, dim=0)
        # head_perm_idx = torch.stack(head_perm_idx_list, dim=0)
        # new_row_perm_idx = torch.stack(new_row_perm_idx_list, dim=0)
        # new_col_perm_idx = torch.stack(new_col_perm_idx_list, dim=0)
        # transpose_matrix_q = torch.stack(transpose_matrix_q_list, dim=0)
        # transpose_matrix_k = torch.stack(transpose_matrix_k_list, dim=0)
        # head_deperm_idx = torch.stack(head_deperm_idx_list, dim=0)
        # new_row_deperm_idx = torch.stack(new_row_deperm_idx_list, dim=0)
        # new_col_deperm_idx = torch.stack(new_col_deperm_idx_list, dim=0)

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list

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
            group_size = row_perm_idx.shape[0] // num_groups
            row_perm_idx_groups_sorted = torch.sort(row_perm_idx.view(num_groups, group_size), dim=1)[0]
            col_perm_idx_groups_sorted = torch.sort(col_perm_idx.view(num_groups, group_size), dim=1)[0]

            transpose_matrix_q = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in row_perm_idx_groups_sorted
            ]).T.contiguous()

            transpose_matrix_k = torch.stack([
                torch.stack([
                    ((g >= j * group_size) & (g < (j + 1) * group_size)).sum()
                    for j in range(num_groups)
                ])
                for g in col_perm_idx_groups_sorted
            ]).T.contiguous()

            new_row_perm_idx = torch.cat([
                row_perm_idx_groups_sorted[(row_perm_idx_groups_sorted >= i * group_size) & (row_perm_idx_groups_sorted < (i + 1) * group_size)]
                for i in range(num_groups)
            ]).reshape(ring_degree, -1)
            # sparse_final = sparse_reordered.index_select(1, new_row_perm_idx.view(-1))
            new_row_perm_idx = new_row_perm_idx - new_row_perm_idx.min(dim=1, keepdim=True)[0]

            new_col_perm_idx = torch.cat([
                col_perm_idx_groups_sorted[(col_perm_idx_groups_sorted >= i * group_size) & (col_perm_idx_groups_sorted < (i + 1) * group_size)]
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

            idx_rows = row_perm_idx_groups_sorted.view(-1).to(sparse_reordered.device, dtype=torch.long)
            idx_cols = col_perm_idx_groups_sorted.view(-1).to(sparse_reordered.device, dtype=torch.long)
            sparse_final = sparse_reordered.index_select(1, idx_rows).index_select(2, idx_cols).contiguous()
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

    return sparse_final, head_perm_idx_list, new_row_perm_idx_list, new_col_perm_idx_list, transpose_matrix_q_list, transpose_matrix_k_list, head_deperm_idx_list, new_row_deperm_idx_list, new_col_deperm_idx_list

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
        hybrid_imbalance_ratio_list = []
        for b in range(batch):
            hybrid_imbalance_ratio_list.append(hybrid_imbalance_ratio(sparse[b], ulysses_degree, ring_degree))
        return sum(hybrid_imbalance_ratio_list) / len(hybrid_imbalance_ratio_list)

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

def create_swat_mask(heads=40, seq_len=1184, window=100):
    """
    Create a sliding-window attention (SWAT) sparse mask.
    
    Args:
        batch_size (int): batch size, default 1
        heads (int): number of attention heads
        seq_len (int): sequence length
        window (int): sliding window size (tokens before/after)
    
    Returns:
        torch.Tensor: [heads, seq_len, seq_len] binary mask
    """
    mask = torch.zeros((heads, seq_len, seq_len), dtype=torch.bool)
    
    for i in range(seq_len):
        start = max(0, i - window)
        end = min(seq_len, i)
        mask[:, i, start:end] = True
    
    return mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test hybrid attention with configurable sequence length"
    )
    parser.add_argument(
        "--seqlen", type=int, default=1184*64, help="sequence length (default: 1024)"
    )
    parser.add_argument(
        "--use_bwd",
        action="store_true",
        help="whether to test backward pass (default: False)",
    )
    parser.add_argument(
        "--sp_ulysses_degree",
        type=int,
        default=None,
        help="sp_ulysses_degree (default: world_size)",
    )
    parser.add_argument(
        "--ring_impl_type",
        type=str,
        default="basic",
        choices=["basic", "zigzag", "basic_flashinfer"],
        help="ring implementation type (default: basic)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="whether to use causal attention (default: False)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="torch",
        choices=[
            "torch",
            "fa",
            "fa3",
            "flashinfer",
            "sage_fp16",
            "sage_fp8",
            "sparse_sage",
            "sage_fp8_sm90",
            "sage_fp16_triton",
            "sage_auto",
            "paro",
            "sparge",
        ],
        help="attention implementation type (default: torch)",
    )
    parser.add_argument(
        "--sparse_sage_l1",
        type=float,
        default=0.07,
        help="l1 for sparse sage attention (default: 0.07)",
    )
    parser.add_argument(
        "--sparse_sage_pv_l1",
        type=float,
        default=0.08,
        help="pv_l1 for sparse sage attention (default: 0.08)",
    )
    parser.add_argument(
        "--sparse_sage_tune_mode",
        action="store_true",
        default=False,
        help="enable tune mode for sparse sage attention (default: False)",
    )
    parser.add_argument(
        "--sparse_sage_tune_path",
        type=str,
        default="./sparsesage_autotune.pt",
        help="path to the sparse sage autotune results (default: ./sparsesage_autotune.pt)",
    )
    return parser.parse_args()


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"[Rank#0] {msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[Rank#{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


# test it with:
# torchrun --nproc_per_node=4  test/test_hybrid_attn.py
if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Inference mainly uses fp16; ROCM flash attention with bf16 precision is slightly larger, will be fixed soon
    dtype = torch.bfloat16 # bfloat in real inference
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = args.seqlen
    nheads = 40
    d = 128
    dropout_p = 0
    causal = args.causal
    deterministic = False

    use_bwd = args.use_bwd

    assert seqlen % world_size == 0
    assert d % 8 == 0

    ring_impl_type = args.ring_impl_type

    sp_ulysses_degree = (
        args.sp_ulysses_degree if args.sp_ulysses_degree is not None else world_size
    )
    sp_ring_degree = world_size // sp_ulysses_degree

    print(
        f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}"
    )

    # Prepare inputs
    q = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )
    data = torch.load("/root/chensiqi/query_key_value_dict.pt", map_location='cpu', weights_only=True)
    q=data['query'].to(device).transpose(1,2).contiguous()[:,:4032*8,:,:]
    k=data['key'].to(device).transpose(1,2).contiguous()[:,:4032*8,:,:]
    v=data['value'].to(device).transpose(1,2).contiguous()[:,:4032*8,:,:]

    # sparse_data = torch.load("/root/chensiqi/sparse_expanded.pth", map_location='cpu', weights_only=True) # 0.4141
    # # sparse_data = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True) # 0.61
    # sparse = sparse_data['sparse'][0].cuda()  # [40, 40, 1182, 1182]
    # H, W = sparse.shape[-2], sparse.shape[-1]
    # pad_h = (8 - H % 8) if H % 8 != 0 else 0
    # pad_w = (8 - W % 8) if W % 8 != 0 else 0
    # if pad_h != 0 or pad_w != 0:
    #     sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
    # sparse=sparse[36].unsqueeze(0).transpose(1, 2).to(device).contiguous()  # [1, 40, H, W]
    sparse = torch.ones((1,504,40,504), device=device, dtype=torch.bool)  # test dense mask

    head_perm_idx =None
    head_deperm_idx =None
    new_row_perm_idx =None
    new_col_perm_idx =None
    new_row_deperm_idx =None
    new_col_deperm_idx =None
    transpose_matrix_q =None
    transpose_matrix_k =None
    
    print(sparse.shape)
    sparse_org = sparse.to(device).contiguous()
    sparse, head_perm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, head_deperm_idx, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v4(sparse.squeeze(0).transpose(0, 1),sp_ulysses_degree,sp_ring_degree,0)
    # sparse, _, new_row_perm_idx, new_col_perm_idx, _, _, _, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v4(sparse.squeeze(0).transpose(0, 1),sp_ulysses_degree,sp_ring_degree,0)

    # sparse, head_perm_idx, new_row_perm_idx, _, transpose_matrix_q, _, head_deperm_idx, new_row_deperm_idx, _ = hybrid_permute_v4(sparse.squeeze(0).transpose(0, 1),sp_ulysses_degree,sp_ring_degree,0)
    # print(sparse.float().sum()/sparse.numel())

    sparse = sparse.unsqueeze(0).transpose(1, 2).to(device).contiguous() #[1, 1182, 40, 1182]
    # print(sparse.equal(sparse_org))
    
    print(f"original imbalance ratio: {hybrid_imbalance_ratio(sparse_org.transpose(1, 2), sp_ulysses_degree, sp_ring_degree)}, after permute: {hybrid_imbalance_ratio(sparse.transpose(1, 2), sp_ulysses_degree, sp_ring_degree)}")
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(sparse, src=0)
    dist.broadcast(sparse_org, src=0)
    dist.broadcast(dout, src=0)
    if head_perm_idx is not None:
        dist.broadcast(head_perm_idx, src=0)
        dist.broadcast(head_deperm_idx, src=0)
    if new_row_perm_idx is not None:
        dist.broadcast(new_row_perm_idx, src=0)
        dist.broadcast(new_row_deperm_idx, src=0)
    if new_col_perm_idx is not None:
        dist.broadcast(new_col_perm_idx, src=0)
        dist.broadcast(new_col_deperm_idx, src=0)
    if transpose_matrix_q is not None:
        dist.broadcast(transpose_matrix_q, src=0)
    if transpose_matrix_k is not None:
        dist.broadcast(transpose_matrix_k, src=0)

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True


    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            q, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    local_k = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            k, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    local_v = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            v, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    if sparse is not None:
        local_sparse = (
            EXTRACT_FUNC_DICT[ring_impl_type](
                sparse, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
            )
            .detach()
            .clone()
        )
    else:
        local_sparse = None

    if sparse_org is not None:
        local_sparse_org = (
            EXTRACT_FUNC_DICT[ring_impl_type](
                sparse_org, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
            )
            .detach()
            .clone()
        )
    else:
        local_sparse_org = None
        
    # print(f"rank:{rank}, local sparse: {local_sparse.shape}, local q: {local_q.shape}")


    if use_bwd:
        local_q.requires_grad = True
        local_k.requires_grad = True
        local_v.requires_grad = True

    # Map argument to AttnType enum
    attn_impl_map = {
        "torch": AttnType.TORCH,
        "fa": AttnType.FA,
        "fa3": AttnType.FA3,
        "flashinfer": AttnType.FLASHINFER,
        "sage_fp16": AttnType.SAGE_FP16,
        "sage_fp8": AttnType.SAGE_FP8,
        "sage_fp8_sm90": AttnType.SAGE_FP8_SM90,
        "sage_fp16_triton": AttnType.SAGE_FP16_TRITON,
        "sage_auto": AttnType.SAGE_AUTO,
        "sparse_sage": AttnType.SPARSE_SAGE,
        "paro": AttnType.PARO,
        "sparge": AttnType.SPARGE,
    }

    if args.attn_impl == "sparse_sage":
        if use_bwd:
            raise RuntimeError("Sparse Sage attention does not support backward pass")
        from spas_sage_attn.autotune import (
            SparseAttentionMeansim,
            load_sparse_attention_state_dict,
        )

        attn_processor = SparseAttentionMeansim(
            l1=args.sparse_sage_l1, pv_l1=args.sparse_sage_pv_l1, tune_pv=True
        )
    else:
        attn_processor = None

    usp_attn = LongContextAttention(
        ring_impl_type=ring_impl_type,
        attn_type=attn_impl_map[args.attn_impl],
        attn_processor=attn_processor,
    )
    
    from xfuser.core.long_ctx_attention import (
        xFuserLongContextAttention,
    )
    hybrid_seq_parallel_attn = xFuserLongContextAttention(
        # use_kv_cache=self.use_long_ctx_attn_kvcache,
        attn_type=AttnType.SPARGE,
    )
    

    if args.attn_impl == "sparse_sage":
        from spas_sage_attn.autotune import (
            SparseAttentionMeansim,
            extract_sparse_attention_state_dict,
        )
        if not args.sparse_sage_tune_mode:
            saved_state_dict = torch.load(
                args.sparse_sage_tune_path + f".rank{dist.get_rank()}"
            )
            load_sparse_attention_state_dict(
                usp_attn, saved_state_dict, multigpu=True, verbose=True
            )
        else:
            # os.environ["sparse_sage_tune_mode"] = "1"
            saved_state_dict = extract_sparse_attention_state_dict(
                usp_attn, verbose=True
            )
            torch.save(
                saved_state_dict, args.sparse_sage_tune_path + f".rank{dist.get_rank()}"
            )

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    # common test parameters
    window_size = (-1, -1)
    alibi_slopes, attn_bias = None, None
    dropout_mask = None

    print(f"before usp attn forward: {local_q.shape}, {local_k.shape} {local_v.shape}, {local_sparse.shape}, {local_sparse_org.shape}, {local_q.abs().mean()}")

    
    # usp attn forward
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(3):
            _ = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse_org,
            )
            torch.cuda.synchronize()
    
        with record_function("usp_attn_forward_org"):
            local_out_ref = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse_org,
            )
            torch.cuda.synchronize()
        
        print("-----------")
        for _ in range(3):
            _ = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse,
                head_perm_idx=head_perm_idx,
                head_deperm_idx=head_deperm_idx,
                new_row_perm_idx = new_row_perm_idx[rank%sp_ring_degree] if new_row_perm_idx is not None else None,
                new_col_perm_idx = new_col_perm_idx[rank%sp_ring_degree] if new_col_perm_idx is not None else None,
                new_row_deperm_idx = new_row_deperm_idx[rank%sp_ring_degree] if new_row_deperm_idx is not None else None,
                transpose_matrix_q = transpose_matrix_q[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_q_T = transpose_matrix_q.T[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_k = transpose_matrix_k[rank%sp_ring_degree].contiguous() if transpose_matrix_k is not None else None,
                transpose_matrix_k_T = transpose_matrix_k.T[rank%sp_ring_degree].contiguous() if transpose_matrix_k is not None else None,
                transpose_matrix_o = transpose_matrix_q.T[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_o_T = transpose_matrix_q[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
            )
            torch.cuda.synchronize()
        with record_function("usp_attn_forward"):
            local_out = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse,
                head_perm_idx=head_perm_idx,
                head_deperm_idx=head_deperm_idx,
                new_row_perm_idx = new_row_perm_idx[rank%sp_ring_degree] if new_row_perm_idx is not None else None,
                new_col_perm_idx = new_col_perm_idx[rank%sp_ring_degree] if new_col_perm_idx is not None else None,
                new_row_deperm_idx = new_row_deperm_idx[rank%sp_ring_degree] if new_row_deperm_idx is not None else None,
                transpose_matrix_q = transpose_matrix_q[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_q_T = transpose_matrix_q.T[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_k = transpose_matrix_k[rank%sp_ring_degree].contiguous() if transpose_matrix_k is not None else None,
                transpose_matrix_k_T = transpose_matrix_k.T[rank%sp_ring_degree].contiguous() if transpose_matrix_k is not None else None,
                transpose_matrix_o = transpose_matrix_q.T[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
                transpose_matrix_o_T = transpose_matrix_q[rank%sp_ring_degree].contiguous() if transpose_matrix_q is not None else None,
            )
    torch.cuda.synchronize()

    # extract local dout
    local_dout = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    max_memory = torch.cuda.max_memory_allocated(device) / (
        1024 * 1024
    )  # Convert to MB
    print(f"[Rank#{rank}] Maximum GPU memory used: {max_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device)  # Reset stats

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses backward:")
        print("#" * 30)

    # usp attn backward
    if use_bwd:
        local_out.backward(local_dout)

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    # if use_bwd:
    #     out_ref.backward(dout)

    dist.barrier()

    # check correctness
    # When checking correctness, use EXTRACT_FUNC_DICT for reference outputs
    # local_out_ref = EXTRACT_FUNC_DICT[ring_impl_type](
    #     out_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    # )
    # local_out_pt_ref = EXTRACT_FUNC_DICT[ring_impl_type](
    #     out_pt_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    # )

    # log("local (rank) out", local_out, rank0_only=True)
    # log("out (distributed) - out_ref (non-distributed) diff", local_out_ref - local_out)
    # log("mean of local out", local_out.mean(), rank0_only=True)
    if rank == 0:
        print(local_out[0,:2,:40,127])
        print(local_out_ref[0,:2,:40,127])
        print(local_out.shape)
        print(local_out_ref.shape)
        print(local_out.abs().mean(),(local_out - local_out_ref).abs().mean())
        # 简单统计：non-zero 元素个数（diff != 0）
        diff = local_out - local_out_ref
        nonzero_cnt = int((diff.abs() > 0).sum().item())
        total = diff.numel()
        print(f"diff != 0 count: {nonzero_cnt}/{total} ({100.0*nonzero_cnt/total:.2f}%)")
    # log("out_ref (non-distributed) - out_pt_ref (gpu) diff", local_out_ref - local_out_pt_ref)

    # torch.testing.assert_close(local_out, local_out_ref, atol=1e-1, rtol=0)
    # torch.testing.assert_close(out_ref, out_pt_ref, atol=1e-2, rtol=0)

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 1 :
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace(f"profile/wan/profile_rank{rank}_ulysses{sp_ulysses_degree}ring{sp_ring_degree}_correctness.json")  # 可选：导出火焰图


# test