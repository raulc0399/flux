import torch
from controlnet_aux import DepthPreprocessor
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

prompt = "Architecture photography of a row of houses with a black railings on the balcony, white exterior, warm sunny day, natural lens flare. the houses are on a private street, surronded by a clean lawn"
control_image = load_image(
    "../imgs/control_images/control_image_depth.png"
)

print(control_image)

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

del processor

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/1",
    device_map="balanced"
)

print(f"Pipeline device map: {pipe.hf_device_map}")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
image.save("output_depth.png")