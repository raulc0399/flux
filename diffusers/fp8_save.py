# from https://huggingface.co/docs/diffusers/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel

import torch
import os
import json

from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize, quantization_map
from safetensors.torch import save_file

output_dir = "./flux-dev-fp8"
bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

os.makedirs(output_dir, exist_ok=True)

print("loading transformer")
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

print("quantize transformer")
quantize(transformer, weights=qfloat8)
freeze(transformer)

print("saving transformer")
save_file_path = f"./{output_dir}/flux1-dev-fp8.safetensors"
save_file(transformer.state_dict(), save_file_path)

save_map_file_path = f"./{output_dir}/flux1-dev-fp8_quantization_map.json"
with open(save_map_file_path, 'w') as f:
    json.dump(quantization_map(transformer), f)

del transformer

print("loading text_encoder_2")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)

print("quantize text_encoder_2")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

print("save text_encoder_2")
save_file_path = f"./{output_dir}/text_encoder_2-fp8.safetensors"
save_file(text_encoder_2.state_dict(), save_file_path)

save_map_file_path = f"./{output_dir}/text_encoder_2-fp8_quantization_map.json"
with open(save_map_file_path, 'w') as f:
    json.dump(quantization_map(text_encoder_2), f)