#!/bin/bash

echo "=============================="
echo " HPC IMAGE GENERATION DEMO"
echo "=============================="

echo ""
echo "Running CPU image generation..."
python image_generate.py --device cpu --out cpu_image.png

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Running GPU image generation..."
    python image_generate.py --device cuda --out gpu_image.png
else
    echo "No NVIDIA GPU detected."
fi

echo ""
echo "Results saved as:"
echo " - cpu_image.png"
echo " - gpu_image.png"
