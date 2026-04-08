import torch
import diffusers
import torch.distributed as dist
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# from wanpipeline import WanPipeline_NEW
# diffusers.pipelines.wan.WanPipeline = WanPipeline_NEW
# from wan import NEW_WanTransformer3DModel
# diffusers.models.WanTransformer3DModel = NEW_WanTransformer3DModel

# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

model_id = "/home/models/Wan2.1-T2V-14B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# print(pipe.transformer.blocks[0].attn1)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_batch_dim_size=1,
        max_ring_dim_size=8,
    ),
)
# 取卡数和max_ring_dim_size的公因数作为ring_degree

# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# apply_cache_on_pipe(pipe,residual_diff_threshold=0.08)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_vae_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

for i in range(2):
    video = pipe(
        prompt="An astronaut dancing vigorously on the moon with earth flying past in the background, hyperrealistic",
        negative_prompt="",
        height=720,#480,
        width=1280,#832,
        num_frames=81,
        num_inference_steps=50,
        guidance_scale=1.0,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]

if dist.get_rank() == 0:
    print("Saving video to results/parallel/wan.mp4")
    export_to_video(video, "results/parallel/test.mp4", fps=15)

dist.destroy_process_group()