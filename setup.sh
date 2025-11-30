#!/bin/bash

# Setup script for AIROGS baseline project
# This script prepares the environment and data structure

echo "=========================================="
echo "AIROGS Baseline - Setup Script"
echo "=========================================="

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/0
mkdir -p outputs/models
mkdir -p outputs/plots
mkdir -p logs

echo "✅ Directories created:"
echo "   - data/0          (place training images here)"
echo "   - outputs/models  (trained models will be saved here)"
echo "   - outputs/plots   (evaluation plots will be saved here)"
echo "   - logs            (training logs)"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed"

# Check TensorFlow installation
echo ""
echo "Checking TensorFlow installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"

# Create data structure info
echo ""
echo "=========================================="
echo "Data Setup Instructions"
echo "=========================================="
echo ""
echo "Please organize your data as follows:"
echo ""
echo "data/"
echo "├── 0/                    (training images)"
echo "│   ├── TRAIN000000.jpg"
echo "│   ├── TRAIN000001.jpg"
echo "│   └── ..."
echo "└── train_labels.csv      (labels file)"
echo ""
echo "The labels CSV should have columns:"
echo "  - challenge_id: image ID (e.g., TRAIN000000)"
echo "  - class: label (RG or NRG)"
echo ""

# Check if data exists
if [ -f "data/train_labels.csv" ]; then
    echo "✅ Found train_labels.csv"

    # Count lines
    num_lines=$(wc -l < data/train_labels.csv)
    echo "   Number of entries: $((num_lines - 1))"
else
    echo "⚠️  train_labels.csv not found in data/"
fi

if [ -d "data/0" ] && [ "$(ls -A data/0)" ]; then
    echo "✅ Found images in data/0/"

    # Count images
    num_images=$(ls data/0/*.jpg 2>/dev/null | wc -l)
    echo "   Number of images: $num_images"
else
    echo "⚠️  No images found in data/0/"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x train_cluster.sh
chmod +x train_cluster_quick.sh
chmod +x setup.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Place your data in the data/ directory"
echo "2. Train locally:"
echo "   python train.py"
echo ""
echo "3. Or submit to cluster:"
echo "   sbatch train_cluster_quick.sh  (quick test)"
echo "   sbatch train_cluster.sh        (full training)"
echo ""
echo "4. Run inference:"
echo "   python inference.py --model outputs/models/your_model.h5 --image path/to/image.jpg"
echo ""
echo "For more information, see README.md"
echo ""

