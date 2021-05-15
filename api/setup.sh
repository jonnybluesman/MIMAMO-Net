#!/bin/bash

# First create the conda environment
conda env create -f environment.yml
# Download the model checkpoint
bash  models/download_models.sh

# Setup repositories and retrieve checkpoint for facial feature extraction
cd .. && git clone https://github.com/albanie/pytorch-benchmarks.git

mkdir pytorch-benchmarks/ferplus/
wget -P pytorch-benchmarks/ferplus/ \
    http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.py
wget -P pytorch-benchmarks/ferplus/ \
    http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.pth
