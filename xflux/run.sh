#!/bin/bash

python3 main.py \
 --use_lora --lora_weight 0.7 \
 --width 1024 --height 768 \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name realism_lora.safetensors \
 --guidance 4 \
 --prompt "contrast play photography of a black female wearing white suit and albino asian geisha female wearing black suit, solid background, avant garde, high fashion"

 python3 main.py \
 --prompt "cyberpank dining room, full hd, cinematic" \
 --image ../imgs/ctrl1024.jpg --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4