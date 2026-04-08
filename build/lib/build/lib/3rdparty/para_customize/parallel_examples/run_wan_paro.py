import torch
import diffusers
import torch.distributed as dist
import time
from diffusers import WanPipeline
from diffusers.utils import export_to_video
import argparse

from wanpipeline import WanPipeline_PARO
diffusers.pipelines.wan.WanPipeline = WanPipeline_PARO
from wan import PARO_WanTransformer3DModel
diffusers.models.WanTransformer3DModel = PARO_WanTransformer3DModel

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

def set_seq_parallel_pg(
    sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True
):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (
        world_size % sp_degree == 0
    ), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    )
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulysses_pg = group

            for i in range(num_ring_pgs):
                ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(
                        i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset
                    )
                )
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs)
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulysses_pg = group

    return ulysses_pg, ring_pg, dp_degree, sp_degree

def parse_args():
    parser = argparse.ArgumentParser(description='Wan video generation with parallel options')
    parser.add_argument('--sp_ulysses_degree', type=int, default=2,
                        help='Ulysses degree for sequence parallelism')
    parser.add_argument('--sp_ring_degree', type=int, default=4,
                        help='Ring degree for sequence parallelism')
    parser.add_argument('--use_db_sp', action='store_true', default=False,
                        help='Use dual-balance sequence parallelism')
    parser.add_argument('--model_id', type=str, 
                        default="/mnt/public/public_models/Wan2.1-T2V-14B-Diffusers",
                        help='Path to model')
    parser.add_argument('--sparse_path', type=str,
                        default="/mnt/public/chensiqi/wan_sparse_mask1.pt",
                        help='Path to sparse data')
    parser.add_argument('--prompt', type=str,
                        default="An astronaut dancing vigorously on the moon with earth flying past in the background, hyperrealistic",
                        help='Generation prompt')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width')
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of frames')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale')
    parser.add_argument('--reward', type=float, default=2,
                        help='Reward parameter for hybrid permute')
    parser.add_argument('--time_log_path', type=str, default="./run_time_log.csv",
                        help='Path to save timing log (rank 0 only)')
    return parser.parse_args()

# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

def main():
    args = parse_args()
    
    dist.init_process_group()

    torch.cuda.set_device(dist.get_rank())

    model_id = args.model_id
    pipe = WanPipeline_PARO.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # flow shift should be 3.0 for 480p images, 5.0 for 720p images
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")

    from para_attn.context_parallel import init_context_parallel_mesh
    from para_attn.context_parallel.diffusers_adapters import parallelize_pipe_paro

    sp_ulysses_degree = args.sp_ulysses_degree
    sp_ring_degree = args.sp_ring_degree

    ulysses_pg, ring_pg, dp_degree, sp_degree = set_seq_parallel_pg(
        sp_ulysses_degree,
        sp_ring_degree,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )

    parallelize_pipe_paro(
        pipe,
        mesh=init_context_parallel_mesh(
            pipe.device.type,
            max_batch_dim_size=1,
            max_ring_dim_size=sp_ring_degree,
        ),
    )

    # hybrid
    sparse_data = torch.load(args.sparse_path, map_location='cpu', weights_only=True)
    sparse = sparse_data.cuda()  # [40, 40, 1182, 1182]

    H, W = sparse.shape[-2], sparse.shape[-1]
    pad_h = (sp_degree - H % sp_degree) if H % sp_degree != 0 else 0
    pad_w = (sp_degree - W % sp_degree) if W % sp_degree != 0 else 0
    if pad_h != 0 or pad_w != 0:
        sparse = torch.nn.functional.pad(sparse, (0, pad_w, 0, pad_h), "constant", 0)
    print(sparse.shape)

    head_perm_idx = None
    head_deperm_idx = None
    new_row_perm_idx = None
    new_col_perm_idx = None
    transpose_matrix_q = None
    transpose_matrix_k = None
    new_row_deperm_idx = None
    new_col_deperm_idx = None

    # if dist.get_rank() == 0:
    #     for block in range(sparse.shape[0]):
    #         print(f"{block}, imbalance ratio:, {hybrid_imbalance_ratio(sparse[block], sp_ulysses_degree, sp_ring_degree)}")

    # 根据use_db_sp参数决定是否进行重排
    if args.use_db_sp:
        sparse, _, new_row_perm_idx, new_col_perm_idx, _, _, _, new_row_deperm_idx, new_col_deperm_idx = hybrid_permute_v4(
            sparse, sp_ulysses_degree, sp_ring_degree, args.reward
        )

    second_timing_sec = None
    second_attention_s = None
    for i in range(2):
        torch.cuda.synchronize()
        start_t = time.perf_counter()
        output = pipe(
            prompt=args.prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="latent",# "pil" if dist.get_rank() == 0 else "pt",
            sparse=sparse,
            head_perm_idx=head_perm_idx if head_perm_idx is not None else None,
            head_deperm_idx=head_deperm_idx if head_deperm_idx is not None else None,
            new_row_perm_idx=new_row_perm_idx if new_row_perm_idx is not None else None,
            new_col_perm_idx=new_col_perm_idx if new_col_perm_idx is not None else None,
            transpose_matrix_q=transpose_matrix_q if transpose_matrix_q is not None else None,
            transpose_matrix_k=transpose_matrix_k if transpose_matrix_k is not None else None,
            new_row_deperm_idx=new_row_deperm_idx if new_row_deperm_idx is not None else None,
            ulysses_pg=ulysses_pg,
            ring_pg=ring_pg,
        )
        # video = output.frames[0]
        torch.cuda.synchronize()
        end_t = time.perf_counter()
        if i == 1:
            second_timing_sec = end_t - start_t
            if dist.get_rank() == 0 and hasattr(output, "attention_time_ms"):
                second_attention_s = output.attention_time_ms / 1000.0

    if dist.get_rank() == 0:
        import csv
        import os
        
        # 直接使用.csv后缀
        csv_file_path = args.time_log_path  # 确保传入的路径已经是.csv后缀
        
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.isfile(csv_file_path)
        
        with open(csv_file_path, "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            
            # 如果文件不存在，先写入表头
            if not file_exists:
                writer.writerow([
                    'model_id', 
                    'num_gpus', 
                    'timing_sec', 
                    'attention_time_s', 
                    'sp_ulysses', 
                    'sp_ring', 
                    'use_db_sp'
                ])
            
            # 写入数据行
            writer.writerow([
                args.model_id,
                dist.get_world_size(),
                f"{second_timing_sec:.3f}",
                f"{second_attention_s:.3f}",
                args.sp_ulysses_degree,
                args.sp_ring_degree,
                args.use_db_sp
            ])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()