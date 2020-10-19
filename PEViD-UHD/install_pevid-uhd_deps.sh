#!/usr/bin/env bash
# Installs necessary dependencies for extracting data from PEViD-UHD dataset

# Update packages list
sudo apt update

# Install FFmpeg using Advanced Packaging Tool (apt)
sudo apt install ffmpeg

# Validate that the package is installed properly
ffmpeg -version

# Install pip packages
pip install Pillow
