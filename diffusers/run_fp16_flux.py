import torch
from diffusers import FluxPipeline
from datetime import datetime

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16) # can replace schnell with dev

# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../imgs/{timestamp}_flux_fp16.png"

out.save(output_path)
