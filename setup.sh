#!/bin/bash

conda env remove --name qMTEB -y
conda create --name qMTEB python=3.9 -y

source activate qMTEB


conda install -c intel openmp 
conda install nomkl 

conda install pytorch torchvision -c pytorch
conda install -c conda-forge sentence-transformers
conda install -c huggingface transformers

pip install mteb

rm -rf results/

source link.sh

echo "Setup completed!"

