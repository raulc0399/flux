import torch
from diffusers import FluxPipeline
from datetime import datetime
# from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight, autoquant
from optimum.quanto import freeze, qfloat8, quantize

prompt = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor. An external black metal staircase leads to the upper entrance, adding visual interest.
The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit. The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced")
# pipe.load_lora_weights("XLabs-AI/flux-RealismLora")
# pipe.fuse_lora(lora_scale=1.1)

# pipe.enable_model_cpu_offload()

# quantize_(pipe.transformer, int8_weight_only())
# quantize_(pipe.text_encoder, int8_weight_only())
# quantize_(pipe.text_encoder_2, int8_weight_only())
# quantize_(pipe.vae, int8_weight_only())

quantize(pipe.transformer, weights=qfloat8)
freeze(pipe.transformer)

quantize(pipe.text_encoder_2, weights=qfloat8)
freeze(pipe.text_encoder_2)

# quantize_(pipe.transformer, int8_dynamic_activation_int8_weight())
# quantize_(pipe.text_encoder, int8_dynamic_activation_int8_weight())
# quantize_(pipe.text_encoder_2, int8_dynamic_activation_int8_weight())
# quantize_(pipe.vae, int8_dynamic_activation_int8_weight())

# pipe.to("cuda")

print(pipe.hf_device_map)

out = pipe(
    prompt=prompt,
    guidance_scale=4,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../imgs/{timestamp}_flux.png"

out.save(output_path)
