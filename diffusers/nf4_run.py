import torch
import gc

from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"
text_encoder_2_4bit = T5EncoderModel.from_pretrained(ckpt_4bit_id, subfolder="text_encoder_2")

ckpt_id = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(ckpt_id, text_encoder_2=text_encoder_2_4bit, transformer=None, vae=None, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt = "armina woman running on a beatufil beach, looking at the camera"
with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
    pipeline = pipeline.to("cpu")
    del pipeline
    
    flush()

transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
pipeline = FluxPipeline.from_pretrained(ckpt_id, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, transformer=transformer_4bit, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

print("Loading LoRA weights.")
pipeline.load_lora_weights("./1.safetensors")
# print("Fusing LoRA.")
# pipeline.fuse_lora()
# print("Unloading LoRA weights.")
# pipeline.unload_lora_weights()

print("Running denoising.")
height, width = 512, 768
images = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, num_inference_steps=50, guidance_scale=5.5, height=height, width=width, output_type="pil").images

images[0].save("output.png")