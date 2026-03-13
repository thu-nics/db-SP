import torch
import diffusers
import torch.distributed as dist
import time
from diffusers import WanPipeline
from diffusers.utils import export_to_video
import argparse
import os

from wanpipeline import WanPipeline_NEW
diffusers.pipelines.wan.WanPipeline = WanPipeline_NEW
from wan import NEW_WanTransformer3DModel
diffusers.models.WanTransformer3DModel = NEW_WanTransformer3DModel

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
    parser = argparse.ArgumentParser(description='Wan video generation with parallel options (NEW version)')
    parser.add_argument('--sp_ulysses_degree', type=int, default=4,
                        help='Ulysses degree for sequence parallelism')
    parser.add_argument('--sp_ring_degree', type=int, default=2,
                        help='Ring degree for sequence parallelism')
    parser.add_argument('--use_db_sp', action='store_true', default=False,
                        help='Use DB sequence parallelism')
    parser.add_argument('--threshold', type=float, default=1.05,
                        help='Threshold for sparse attention')
    parser.add_argument('--use_ulysses_low', action='store_true', default=True,
                        help='Use ulysses low configuration')
    parser.add_argument('--model_id', type=str, 
                        default="/mnt/public/public_models/Wan2.1-T2V-14B-Diffusers",
                        help='Path to model')
    parser.add_argument('--prompt', type=str,
                        default="An astronaut dancing vigorously on the moon with earth flying past in the background, hyperrealistic",
                        help='Generation prompt')
    parser.add_argument('--negative_prompt', type=str, default="",
                        help='Negative prompt')
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
    parser.add_argument('--fps', type=int, default=15,
                        help='FPS for output video')
    parser.add_argument('--time_log_path', type=str, default="./run_time_log.csv",
                        help='Path to save timing log (rank 0 only)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化分布式进程组
    dist.init_process_group()

    torch.cuda.set_device(dist.get_rank())

    model_id = args.model_id
    pipe = WanPipeline_NEW.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # flow shift should be 3.0 for 480p images, 5.0 for 720p images
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")

    from para_attn.context_parallel import init_context_parallel_mesh
    from para_attn.context_parallel.diffusers_adapters import parallelize_pipe_sparge

    sp_ulysses_degree = args.sp_ulysses_degree
    sp_ring_degree = args.sp_ring_degree

    ulysses_pg, ring_pg, dp_degree, sp_degree = set_seq_parallel_pg(
        sp_ulysses_degree,
        sp_ring_degree,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        use_ulysses_low=args.use_ulysses_low,
    )

    parallelize_pipe_sparge(
        pipe,
        mesh=init_context_parallel_mesh(
            pipe.device.type,
            max_batch_dim_size=1,
            max_ring_dim_size=sp_ring_degree,
        ),
    )
    

    second_timing_sec = None
    second_attention_s = None
    count=0
    for i in range(2):
        torch.cuda.synchronize()
        start_t = time.perf_counter()
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ulysses_pg=ulysses_pg, 
            ring_pg=ring_pg,
            use_db_sp=args.use_db_sp,
            threshold=args.threshold,
            output_type="latent",
        )
        torch.cuda.synchronize()
        end_t = time.perf_counter()
        if i == 1:
            second_timing_sec = end_t - start_t
            if dist.get_rank() == 0 and hasattr(output, "attention_time_ms"):
                second_attention_s = output.attention_time_ms / 1000.0
            if dist.get_rank() == 0 and hasattr(output, "attention_time_ms"):
                count = output.count

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
        count_csv_path = args.time_log_path.replace('.csv', '_count.csv') if args.time_log_path.endswith('.csv') else args.time_log_path + '_count.csv'

        count_file_exists = os.path.isfile(count_csv_path)
        
        with open(count_csv_path, "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            
            if not count_file_exists:
                writer.writerow([
                    'threshold',
                    'count'
                ])
            
            writer.writerow([
                args.threshold,
                count
            ])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()