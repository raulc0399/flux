import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from datetime import datetime
import itertools

MODELS = [
    "Xlabs-AI/flux-controlnet-canny-diffusers",
    "Xlabs-AI/flux-controlnet-depth-diffusers",
    "jasperai/Flux.1-dev-Controlnet-Depth",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    "promeai/FLUX.1-controlnet-lineart-promeai",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    "InstantX/FLUX.1-dev-controlnet-canny"
]

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
GENERATOR = torch.Generator(device="cuda").manual_seed(87544357)

PROMPT = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor.
An external black metal staircase leads to the upper entrance, adding visual interest. The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit.
The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""
    
PROMPT1 = """Modern minimalist townhouses at sunset, featuring clean white cubic architecture with black metal staircases and railings. The buildings are arranged in a row on a gentle grassy slope. 
Warm evening light casts long shadows across the facades, with illuminated windows creating a cozy glow. The foreground shows wild flowers and ornamental grasses, including white dandelions and pampas grass.
Two people are visible: one climbing the exterior stairs and a child playing in the garden. The scene has a dreamy, nostalgic quality with soft natural lighting and lens flare effects.
Architectural visualization style with photorealistic rendering, shallow depth of field, and warm color grading."""
    
PROMPT2 = """Modern white townhouses arranged on a hillside at sunset. Minimalist cubic architecture with black metal staircases and balconies. Warm glowing windows and wild grasses with dandelions in the foreground.
Natural lens flare and soft evening lighting. Architectural visualization style with photorealistic rendering."""


def get_control_image(model_name):
    """Select appropriate control image based on model name"""
    control_images = {
        'depth': load_image("../imgs/control_image_depth.png"),
        'canny': load_image("../imgs/control_image_edges.png"),
        'normals': load_image("../imgs/control_image_normals.png")
    }
    
    if 'depth' in model_name.lower():
        return control_images['depth']
    elif 'canny' in model_name.lower():
        return control_images['canny']
    elif 'lineart' in model_name.lower():
        return control_images['canny']
    elif 'normals' in model_name.lower():
        return control_images['normals']
    else:
        print(f"Unknown control image for model: {model_name}")
        return None

def load_pipeline(controlnet_model):
    """Load the pipeline with specified controlnet model"""
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model, 
        torch_dtype=torch.bfloat16
    ).to("cuda:1")

    pipe = FluxControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=torch.bfloat16, 
        device_map="balanced"
    )
    
    print(f"Loaded model: {controlnet_model}")
    print(f"Pipeline device map: {pipe.hf_device_map}")
    print(f"Controlnet device: {controlnet.device}")
    
    return pipe

def generate_image(pipe, control_image, prompt_text, conditioning_scale, num_steps, guidance_scale):
    """Generate image with specified parameters"""
    width, height = control_image.size
    
    image = pipe(
        prompt_text,
        control_image=control_image,
        width=width,
        height=height,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=GENERATOR,
    ).images[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = pipe.controlnet._name_or_path.split('/')[-1]
    prompt_index = PROMPTS.index(prompt_text)
    output_path = f"../imgs/{timestamp}_{model_name}_p{prompt_index}_c{conditioning_scale}_s{num_steps}_g{guidance_scale}.png"
    
    image.save(output_path)
    print(f"Saved: {output_path}")

def main():
    # Parameter combinations
    prompts = [PROMPT, PROMPT1, PROMPT2]
    conditioning_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    inference_steps = [20, 30, 40]
    guidance_scales = [3.5, 4.0]
    
    for model in MODELS:
        try:
            pipe = load_pipeline(model)
            control_image = get_control_image(model)
            
            # Generate all parameter combinations
            params = itertools.product(
                prompts,
                conditioning_scales,
                inference_steps,
                guidance_scales
            )
            
            for prompt_text, cond_scale, steps, guidance in params:
                try:
                    generate_image(
                        pipe,
                        control_image,
                        prompt_text,
                        cond_scale,
                        steps,
                        guidance
                    )
                except Exception as e:
                    print(f"Error generating image for {model} with params: {cond_scale}, {steps}, {guidance}")
                    print(f"Error: {str(e)}")
                    
        except Exception as e:
            print(f"Error loading model {model}")
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
