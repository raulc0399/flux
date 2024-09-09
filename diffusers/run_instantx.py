import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel
from datetime import datetime

control_image = load_image("../imgs/ctrl1024.jpg")
prompt = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor. An external black metal staircase leads to the upper entrance, adding visual interest.
The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit. The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""

base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("XLabs-AI/flux-RealismLora")
pipe.fuse_lora(lora_scale=1.1)
pipe.to("cuda")

image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28,
    guidance_scale=3.5,
    joint_attention_kwargs={"scale": 1}
).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../imgs/{timestamp}_instantx.png"

image.save(output_path)
