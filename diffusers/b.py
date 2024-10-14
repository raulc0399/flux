# First, in your terminal.
#
# $ python3 -m virtualenv env
# $ source env/bin/activate
# $ pip install torch torchvision transformers sentencepiece protobuf accelerate
# $ pip install git+https://github.com/huggingface/diffusers.git
# $ pip install optimum-quanto

import torch

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16

# schnell is the distilled turbo model. For the CFG distilled model, use:
# bfl_repo = "black-forest-labs/FLUX.1-dev"
# revision = "refs/pr/3"
#
# The undistilled model that uses CFG ("pro") which can use negative prompts
# was not released.
bfl_repo = "black-forest-labs/FLUX.1-dev"

print("1")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(bfl_repo, subfolder="tokenizer", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)
print("2")
# Experimental: Try this to load in 4-bit for <16GB cards.
#
# from optimum.quanto import qint4
# quantize(transformer, weights=qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
# freeze(transformer)
quantize(transformer, weights=qfloat8)
freeze(transformer)
print("4")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)
print("5")
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()

generator = torch.Generator().manual_seed(12345)
image = pipe(
    prompt='nekomusume cat girl, digital painting', 
    width=1024,
    height=1024,
    num_inference_steps=4, 
    generator=generator,
    guidance_scale=3.5,
).images[0]
image.save('test_flux_distilled.png')