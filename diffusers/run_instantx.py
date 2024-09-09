import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel

base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"

control_image = load_image("../imgs/ctrl1024.jpg")
prompt = "A girl in city, 25 years old, cool, futuristic"

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
image.save("flux.png")