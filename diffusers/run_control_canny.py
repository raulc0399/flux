import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

prompt = "Architecture photography of a row of houses with a black railings on the balcony, white exterior, warm sunny day, natural lens flare. the houses are on a private street, surronded by a clean lawn"
control_image = load_image(
    "../imgs/control_images/control_image_edges.png"
)

print(control_image)

processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

del processor

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Canny-dev",
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
    guidance_scale=4.5,
).images[0]
image.save("output_canny.png")