import torch
import gc

from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel
from datetime import datetime

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"
ckpt_id = "black-forest-labs/FLUX.1-dev"

text_encoder_2_4bit = T5EncoderModel.from_pretrained(ckpt_4bit_id, subfolder="text_encoder_2")
transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")

pipeline = FluxPipeline.from_pretrained(ckpt_id, text_encoder_2=text_encoder_2_4bit, transformer=transformer_4bit, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

print("Loading LoRA weights.")
pipeline.load_lora_weights("./loras/009.safetensors")
# print("Fusing LoRA.")
# pipeline.fuse_lora()
# print("Unloading LoRA weights.")
# pipeline.unload_lora_weights()

prompt = "cartoon of woman running on a beautiful beach, looking at the camera"
print("Running denoising.")
height, width = 512, 768
with torch.no_grad():
    images = pipeline(prompt=prompt, 
                      num_inference_steps=50, guidance_scale=5.5,
                      height=height, width=width, output_type="pil").images

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
images[0].save(f"./output/{timestamp}_output.png")