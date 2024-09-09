#!/bin/bash

PROMPT='The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor. An external black metal staircase leads to the upper entrance, adding visual interest.
The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit. The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings.'

output_path="../imgs/xflux.png"

 python3 main.py \
 --prompt "$PROMPT" \
 --image ../imgs/ctrl1024.jpg \
 --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4 \
 --use_lora --lora_weight 0.7 \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name realism_lora.safetensors \
 --save_path "$output_path"
