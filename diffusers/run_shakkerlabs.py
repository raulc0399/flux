import torch
from diffusers.utils import load_image

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from datetime import datetime

control_image_canny = load_image("../images/ctrl1024.jpg")
control_mode_canny = 0

prompt = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor. An external black metal staircase leads to the upper entrance, adding visual interest.
The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit. The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

width, height = control_mode_canny.size

image = pipe(
    prompt, 
    control_image=[control_image_canny],
    control_mode=[control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.4],
    num_inference_steps=24, 
    guidance_scale=3.5,
    generator=torch.manual_seed(42),
).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../imgs/{timestamp}_shakkerlabs.png"

image.save(output_path)
