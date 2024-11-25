import os
import torch
from controlnet_aux import DepthPreprocessor
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from pathlib import Path

# Global configuration
BASE_MODEL = "black-forest-labs/FLUX.1-Depth-dev"
INPUT_DIR = Path("../imgs/control_images")
OUTPUT_DIR = Path("../imgs/flux-depth")
PROMPT = "Architecture photography of a row of houses with a black railings on the balcony, white exterior, warm sunny day, natural lens flare. the houses are on a private street, surronded by a clean lawn"

def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

def process_image(input_path, output_dir, processor, pipe):
    # Load and process image
    control_image = load_image(input_path)
    print(f"Processing {input_path}")
    
    # Apply Depth detection
    depth_image = processor(control_image)[0].convert("RGB")
    
    # Save Depth output
    input_name = Path(input_path).stem
    depth_output_path = output_dir / f"{input_name}_depth.png"
    depth_image.save(depth_output_path)
    
    # Generate image with pipeline
    generated_image = pipe(
        prompt=PROMPT,
        control_image=depth_image,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
    ).images[0]
    
    # Save generated image
    output_path = output_dir / f"{input_name}_generated.png"
    generated_image.save(output_path)

def main():
    # Create output directory
    output_dir = ensure_output_dir()
    
    # Initialize processor and pipeline
    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    pipe = FluxControlPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        revision="refs/pr/1",
        device_map="balanced"
    )
    print(f"Pipeline device map: {pipe.hf_device_map}")
    
    # Process all images
    for i in range(1, 7):
        input_path = INPUT_DIR / f"{i}.jpg"
        if input_path.exists():
            process_image(input_path, output_dir, processor, pipe)
    
    del processor, pipe

if __name__ == "__main__":
    main()
