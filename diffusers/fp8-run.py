# from https://huggingface.co/docs/diffusers/en/api/pipelines/flux#single-file-loading-for-the-fluxtransformer2dmodel

import torch
from os import mkdir

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

model_dir = "./flux-dev-fp8"
bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

# load the transformer
flux_dev_path = f"./{model_dir}/flux1-dev-fp8.pt"

config = FluxTransformer2DModel.load_config(bfl_repo, subfolder = "transformer")
transformer = FluxTransformer2DModel.from_config(config).to(dtype)

state_dict = torch.load(flux_dev_path)
transformer.load_state_dict(state_dict, assign = True)

# load the text_encoder_2 models
t5_path = f"./{model_dir}/text_encoder_2-fp8.pt"

config = T5EncoderModel.load_config(bfl_repo, subfolder = "text_encoder_2")
text_encoder_2 = T5EncoderModel.from_config(config).to(dtype)

state_dict = torch.load(t5_path)
text_encoder_2.load_state_dict(state_dict, assign = True)

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
