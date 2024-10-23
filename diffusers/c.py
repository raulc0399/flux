import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

# https://huggingface.co/hf-internal-testing/flux.1-dev-nf4-pkg/discussions/2

# https://colab.research.google.com/github/camenduru/flux-jupyter/blob/main/flux.1-dev_jupyter.ipynb
# https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors
# https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft
# https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors
# https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors

# https://huggingface.co/Kijai/flux-fp8/tree/main

dtype = torch.bfloat16

# https://huggingface.co/Comfy-Org/flux1-dev/tree/main
pipe = FluxPipeline.from_single_file("/home/raul/codelab/models/flux1-dev-fp8.safetensors", torch_dtype=dtype, device_map="auto").to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=20,
    #generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-fp8-dev.png")
