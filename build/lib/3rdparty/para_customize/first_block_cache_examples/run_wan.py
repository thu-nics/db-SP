import torch
import diffusers
from diffusers import WanPipeline

# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from wan import NEW_WanTransformer3DModel
# Special for wan
diffusers.models.WanTransformer3DModel = NEW_WanTransformer3DModel

model_id = "/home/models/Wan2.1-T2V-14B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# apply_cache_on_pipe(pipe,residual_diff_threshold=0.08)

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling()

# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

import ipdb;ipdb.set_trace()

video = pipe(
    prompt="An astronaut dancing vigorously on the moon with earth flying past in the background, hyperrealistic",
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=30,
).frames[0]

print("Saving video to wan.mp4")
export_to_video(video, "wan.mp4", fps=15)
