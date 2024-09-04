#!/bin/bash

 python3 main.py \
 --prompt "cyberpank dining room, full hd, cinematic" \
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
 --lora_name realism_lora.safetensors