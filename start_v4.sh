#!/bin/bash
# V4 QUICK START GUIDE
# ====================

echo "========================================"
echo "V4 ADVANCED PREPROCESSING - QUICK START"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -d "v4_advanced" ]; then
    echo "❌ Error: Run this from MiccaiChallenge directory"
    exit 1
fi

echo "📋 Pre-flight checklist:"
echo ""

# 1. Check V4 files exist
echo "1. Checking V4 files..."
required_files=(
    "v4_advanced/config_v4.py"
    "v4_advanced/dataset_v4.py"
    "v4_advanced/train_v4.py"
    "v4_advanced/evaluate_v4.py"
    "v4_advanced/advanced_preprocessing.py"
    "submit_train_v4.sh"
)

all_present=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        all_present=false
    fi
done

if [ "$all_present" = false ]; then
    echo ""
    echo "❌ Some files are missing. Cannot proceed."
    exit 1
fi

# 2. Check datasets
echo ""
echo "2. Checking datasets..."
dataset_dirs=("dataset/0" "dataset/1" "dataset/4" "dataset/5")
for dir in "${dataset_dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.jpg" 2>/dev/null | wc -l)
        echo "   ✅ $dir ($count images)"
    else
        echo "   ⚠️  $dir (not found)"
    fi
done

# 3. Check Python syntax
echo ""
echo "3. Validating Python syntax..."
cd v4_advanced
if python -m py_compile config_v4.py dataset_v4.py train_v4.py 2>/dev/null; then
    echo "   ✅ All Python files valid"
else
    echo "   ❌ Syntax errors found"
    cd ..
    exit 1
fi
cd ..

# 4. Check SLURM availability
echo ""
echo "4. Checking SLURM..."
if command -v sbatch &> /dev/null; then
    echo "   ✅ SLURM available"
else
    echo "   ⚠️  SLURM not available (cannot submit to cluster)"
fi

echo ""
echo "========================================"
echo "✅ PRE-FLIGHT CHECK PASSED"
echo "========================================"
echo ""

# Show submission command
echo "🚀 To submit V4 training to cluster:"
echo ""
echo "   sbatch submit_train_v4.sh"
echo ""
echo "📊 To monitor training:"
echo ""
echo "   tail -f train_run/output/airogs_v4_*.out"
echo ""
echo "🔍 To check preprocessing samples (after 5 min):"
echo ""
echo "   ls -lh v4_advanced/preprocessing_samples/"
echo ""
echo "📈 To evaluate after training:"
echo ""
echo "   python v4_advanced/evaluate_v4.py v4_advanced/models/*_best.keras --tta"
echo ""
echo "========================================"
echo ""

# Ask if user wants to submit now
read -p "Submit V4 training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Submitting V4 training job..."
    sbatch submit_train_v4.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Job submitted successfully!"
        echo ""
        echo "Check status with: squeue -u $USER"
        echo "Monitor output with: tail -f train_run/output/airogs_v4_*.out"
    else
        echo ""
        echo "❌ Job submission failed"
    fi
else
    echo ""
    echo "Job not submitted. Run manually when ready:"
    echo "   sbatch submit_train_v4.sh"
fi

echo ""
