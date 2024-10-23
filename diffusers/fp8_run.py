# from https://huggingface.co/docs/diffusers/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel

import torch
import json

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from safetensors.torch import load_file
from optimum.quanto import requantize

model_dir = "./flux-dev-fp8"
bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

# load the transformer
flux_dev_path = f"./{model_dir}/flux1-dev-fp8.safetensors"
flux_dev_map_file_path = f"./{model_dir}/flux1-dev-fp8_quantization_map.json"

with torch.device('meta'):
    config = FluxTransformer2DModel.load_config(bfl_repo, subfolder = "transformer")
    transformer = FluxTransformer2DModel.from_config(config).to(dtype)

state_dict = load_file(flux_dev_path)
with open(flux_dev_map_file_path, 'r') as f:
    quantization_map = json.load(f)
requantize(transformer, state_dict, quantization_map, device=torch.device('cuda'))

# load the text_encoder_2 models
t5_path = f"./{model_dir}/text_encoder_2-fp8.saftensors"
t5_map_file_path = f"./{model_dir}/text_encoder_2-fp8_quantization_map.json"

with torch.device('meta'):
    config = T5EncoderModel.load_config(bfl_repo, subfolder = "text_encoder_2")
    text_encoder_2 = T5EncoderModel.from_config(config).to(dtype)

state_dict = load_file(t5_path)
with open(t5_map_file_path, 'r') as f:
    quantization_map = json.load(f)
requantize(text_encoder_2, state_dict, quantization_map, device=torch.device('cuda'))

pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-fp8-dev.png")
