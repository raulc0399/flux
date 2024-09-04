#!/bin/bash

apt-get update
apt-get install -y git

git clone https://github.com/XLabs-AI/x-flux.git

cd x-flux

python3 -m venv xflux_env
source xflux_env/bin/activate

pip install -r requirements.txt
