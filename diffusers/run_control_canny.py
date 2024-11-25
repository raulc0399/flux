import os
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from pathlib import Path

# Global configuration
BASE_MODEL = "black-forest-labs/FLUX.1-Canny-dev"
INPUT_DIR = Path("../imgs/control_images")
OUTPUT_DIR = Path("../imgs/flux-canny")
INFERENCE_STEPS = [30, 40, 50]
GUIDANCE_SCALES = [4, 7, 10, 30]
PROMPT = "Modern minimalist house with sleek geometric designs. Large glass windows and sliding doors integrated into the architecture, featuring wood, white stucco, and dark metal finishes. Houses include clean lines, flat or slightly angled roofs, and landscaped surroundings with wooden decks, patios, or modern walkways. Emphasize contemporary lighting, open spaces, and a harmonious blend of natural materials and modern aesthetic"

def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

def process_image(input_path, output_dir, processor, pipe):
    print(f"Processing {input_path}")

    # Load and process image
    control_image = load_image(input_path)
        
    # Apply Canny detection
    canny_image = processor(
        control_image, 
        low_threshold=50, 
        high_threshold=200, 
        detect_resolution=1024, 
        image_resolution=1024
    )
    
    # Save Canny output
    input_name = Path(input_path).stem
    canny_output_path = output_dir / f"{input_name}_canny.png"
    canny_image.save(canny_output_path)
    
    # Generate images with different parameters
    for steps in INFERENCE_STEPS:
        for guidance in GUIDANCE_SCALES:
            # Generate image with pipeline
            generated_image = pipe(
                prompt=PROMPT,
                control_image=canny_image,
                height=1024,
                width=1024,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]
            
            # Save generated image with parameters in filename
            output_path = output_dir / f"{input_name}_steps{steps}_guidance{guidance}_generated.png"
            generated_image.save(output_path)

def main():
    # Create output directory
    output_dir = ensure_output_dir()
    
    # Initialize processor and pipeline
    processor = CannyDetector()
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
            process_image(str(input_path), output_dir, processor, pipe)
    
    del processor, pipe

if __name__ == "__main__":
    main()
