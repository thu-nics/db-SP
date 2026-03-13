import torch
import diffusers
import torch.distributed as dist
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from cogvideopipeline import CogVideoXPipeline_NEW
diffusers.pipelines.cogvideo.CogVideoXPipeline = CogVideoXPipeline_NEW
from cogvideo import NEW_CogVideoXTransformer3DModel
diffusers.models.CogVideoXTransformer3DModel = NEW_CogVideoXTransformer3DModel

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = CogVideoXPipeline_NEW.from_pretrained(
    "/home/models/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_batch_dim_size=1,
    ),
)

# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# apply_cache_on_pipe(pipe,residual_diff_threshold=0.08)

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=161,
    guidance_scale=1.0,
    # height=768,
    # width=1360,
    # generator=torch.Generator(device=pipe.device).manual_seed(42),
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to cogvideo_test.mp4")
    export_to_video(video, "results/parallel/cogvideo_test.mp4", fps=8)

dist.destroy_process_group()
