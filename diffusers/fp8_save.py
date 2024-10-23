# from https://huggingface.co/docs/diffusers/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel

import torch
import os

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

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
save_file_path = f"./{output_dir}/flux1-dev-fp8.pt"
torch.save(transformer.state_dict(), save_file_path)

del transformer

print("loading text_encoder_2")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)

print("quantize text_encoder_2")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

print("save text_encoder_2")
save_file_path = f"./{output_dir}/text_encoder_2-fp8.pt"
torch.save(text_encoder_2.state_dict(), save_file_path)
