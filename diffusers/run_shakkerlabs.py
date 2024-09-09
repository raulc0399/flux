import torch
from diffusers.utils import load_image

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'

control_image_canny = load_image("../images/ctrl1024.jpg")
control_mode_canny = 0

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
