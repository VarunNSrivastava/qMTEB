#!/bin/bash

conda env remove --name qMTEB -y
conda create --name qMTEB python=3.9 -y

source activate qMTEB

conda install -c intel openmp 
conda install nomkl 
pip install torch torchvision torchaudio
pip install  -e /Users/varun/documents/python/embeddings/sentence-transformers
pip install mteb
pip install onnxruntime-silicon
python -m pip install "optimum[onnxruntime]@git+https://github.com/huggingface/optimum.git"

source link.sh

source activate qMTEB

echo "Setup completed!"

