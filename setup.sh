#!/bin/bash
set -e

cd outfit-transformer
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda for the current shell session
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Create and activate the conda environment
conda create -n outfit-transformer python=3.12.4 -y
conda activate outfit-transformer

# Update environment from environment.yml file
conda env update -f environment.yml

# Install unzip using apt and gdown using pip
apt update && apt install unzip -y
pip install gdown

# Download and extract the polyvore dataset
mkdir -p datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu -O polyvore.zip
unzip polyvore.zip -d ./datasets/polyvore
rm polyvore.zip

# Download and extract checkpoints
mkdir -p checkpoints
gdown --id 1mzNqGBmd8UjVJjKwVa5GdGYHKutZKSSi -O checkpoints.zip
unzip checkpoints.zip -d ./checkpoints
rm checkpoints.zip

# Run the Python script for generating CLIP embeddings
python -m src.run.1_generate_clip_embeddings
