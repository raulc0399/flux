import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel
from datetime import datetime
import itertools
import json
import os

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
UNION_MODELS = [
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "InstantX/FLUX.1-dev-Controlnet-Union"
]
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

PROMPT3="""Architecture photography of a row of houses with a black railings on the balcony, white exterior, warm sunny day, natural lens flare. the houses are on a private street, surronded by a clean lawn"""

def get_control_images():
    """Load control images"""
    return {
        'depth': load_image("../imgs/control_images/control_image_depth.png"),
        'canny': load_image("../imgs/control_images/control_image_edges.png")
    }

def load_pipeline(model_name):
    """Load the pipeline with union controlnet model"""
    controlnet_union = FluxControlNetModel.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16
    ).to("cuda:1")
    
    controlnet = FluxMultiControlNetModel([controlnet_union])

    pipe = FluxControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=torch.bfloat16, 
        device_map="balanced"
    )
    
    print(f"Loaded model: {model_name}")
    print(f"Pipeline device map: {pipe.hf_device_map}")
    print(f"Controlnet device: {controlnet.device}")
    
    return pipe

def generate_image(pipe, control_images, prompt_text, control_modes, conditioning_scale, num_steps,
                   guidance_scale, image_index, model_name):
    """Generate image with specified parameters"""
    width, height = control_images['depth'].size
    
    # Convert single conditioning scale to list if needed
    if isinstance(conditioning_scale, (int, float)):
        conditioning_scale = [conditioning_scale] * len(control_modes)
    
    # Prepare control images and modes based on modes list
    control_imgs = []
    for mode in control_modes:
        if mode == 2:  # depth
            control_imgs.append(control_images['depth'])
        elif mode == 0:  # canny
            control_imgs.append(control_images['canny'])
    
    image = pipe(
        prompt_text,
        control_image=control_imgs,
        control_mode=control_modes,
        width=width,
        height=height,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=GENERATOR,
        # joint_attention_kwargs={"scale": 1}
    ).images[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{image_index:04d}_c{conditioning_scale}_s{num_steps}_g{guidance_scale}"
    
    # Save image
    image_path = f"../imgs/{model_name}/{base_name}.png"
    image.save(image_path)
    
    # Save parameters
    params = {
        "model_name": model_name,
        "control_modes": control_modes,
        "conditioning_scale": conditioning_scale,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "image_path": image_path,
        "control_images": ["control_image_depth.png" if mode == 2 else "control_image_edges.png" for mode in control_modes],
        "prompt": prompt_text
    }
    
    params_path = f"../imgs/{model_name}/params/{base_name}.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4, separators=(',\n', ': '))
    
    print(f"Saved image: {image_path}")

def ensure_params_dir(model):
    params_dir = f"../imgs/{model}/params"
    os.makedirs(params_dir, exist_ok=True)

def main(model_index):
    ensure_params_dir()

    image_index = 0

    # Define all parameter combinations
    prompts = [PROMPT3]
    base_configs = [
        {'modes': [2]},           # depth only
        {'modes': [0]},           # canny only
        {'modes': [2, 0]},        # both controls
        {'modes': [0, 2]},        # both controls
    ]
    conditioning_scales = [0.6, 0.8, 1.0]
    inference_steps = [30, 40]
    guidance_scales = [3.5, 4.0]

    # Calculate total combinations
    total_combinations = (
        len(prompts) * 
        len(base_configs) * 
        len(conditioning_scales) * 
        len(inference_steps) * 
        len(guidance_scales)
    )
    print(f"Total combinations to generate: {total_combinations}")

    # Generate all parameter combinations using itertools
    param_combinations = itertools.product(
        prompts,
        base_configs,
        conditioning_scales,
        inference_steps,
        guidance_scales
    )

    control_images = get_control_images()
    
    model = UNION_MODELS[model_index]
    try:
        union_model = model.replace("/", "-")
        ensure_params_dir(union_model)

        pipe = load_pipeline(model)

        for prompt, config, scale, steps, guidance in param_combinations:
            try:
                # For dual control, use same scale for both
                if len(config['modes']) > 1:
                    config_scale = [scale] * len(config['modes'])
                else:
                    config_scale = scale
                    
                generate_image(
                    pipe,
                    control_images,
                    prompt,
                    config['modes'],
                    config_scale,
                    num_steps=steps,
                    guidance_scale=guidance,
                    image_index=image_index
                )

                image_index += 1

            except Exception as e:
                print(f"Error generating image for config: {config}")
                print(f"Error: {str(e)}")
                                
    except Exception as e:
        print(f"Error loading model {union_model}")
        print(f"Error: {str(e)}")

def check_model_index(index) -> bool:
    try:
        index = int(index)
        if 0 <= index < len(MODELS):
            return True
        else:
            return False
    except ValueError:
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single.py <model index>")
        exit()
    
    model_index = sys.argv[1]

    if check_model_index(model_index):
        main(int(model_index))
    else:
        print("Index not valid")
