#!/bin/bash
# Update the system
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y build-essential dkms

# Install CUDA Toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# Verify installation
nvidia-smi
