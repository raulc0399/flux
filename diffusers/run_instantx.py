import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel
from datetime import datetime
from accelerate.utils import compute_module_sizes
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight,  autoquant

control_image = load_image("../imgs/ctrl1024.jpeg")
prompt = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor. An external black metal staircase leads to the upper entrance, adding visual interest.
The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit. The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""

base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16).to("cuda:1")
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16, device_map="balanced"
)

# from https://github.com/sayakpaul/diffusers-torchao?tab=readme-ov-file#training-with-fp8
# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = autoquant(torch.compile(pipe.transformer, mode='max-autotune', fullgraph=True), error_on_unseen=False)

# https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47
# quantize_(pipe.transformer, int8_weight_only())

# quantize_(pipe.transformer, int8_dynamic_activation_int8_weight())

pipe.load_lora_weights("XLabs-AI/flux-RealismLora")
pipe.fuse_lora(lora_scale=1.1)
# pipe.to("cuda")
# pipe.enable_model_cpu_offload()

print(pipe.hf_device_map)
print(controlnet.device);
print(compute_module_sizes(controlnet, dtype=torch.bfloat16)[""])

image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=30,
    guidance_scale=4,
    joint_attention_kwargs={"scale": 1}
).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../imgs/{timestamp}_instantx.png"

image.save(output_path)
