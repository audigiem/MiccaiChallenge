#!/bin/bash
#
# Verification script - checks all configurations are correct for RTX 6000
#

echo "============================================================"
echo "RTX 6000 Configuration Verification"
echo "============================================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Training script partition
echo "🔍 Checking training script..."
if grep -q "#SBATCH --partition=rtx6000" submit_train_improved.sh; then
    echo "   ✅ Partition: rtx6000"
else
    echo "   ❌ ERROR: Partition not set to rtx6000"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Evaluation script partition
echo ""
echo "🔍 Checking evaluation script..."
if grep -q "#SBATCH --partition=rtx6000" submit_eval_improved.sh; then
    echo "   ✅ Partition: rtx6000"
else
    echo "   ❌ ERROR: Partition not set to rtx6000"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Config batch size
echo ""
echo "🔍 Checking config_improved.py..."
if grep -q "BATCH_SIZE = 16" config_improved.py; then
    echo "   ✅ Batch size: 16"
else
    echo "   ⚠️  WARNING: Batch size not 16"
    WARNINGS=$((WARNINGS + 1))
fi

if grep -q "USE_MIXED_PRECISION = False" config_improved.py; then
    echo "   ✅ Mixed precision: Disabled"
else
    echo "   ⚠️  WARNING: Mixed precision not disabled"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 4: Evaluation test split
echo ""
echo "🔍 Checking evaluation_improved.py..."
if grep -q "train_split=0.0" evaluation_improved.py && grep -q "test_split=1.0" evaluation_improved.py; then
    echo "   ✅ Test split: All data (same as baseline)"
else
    echo "   ❌ ERROR: Test split not configured correctly"
    ERRORS=$((ERRORS + 1))
fi

# Check 5: Scripts are executable
echo ""
echo "🔍 Checking script permissions..."
if [ -x submit_train_improved.sh ]; then
    echo "   ✅ submit_train_improved.sh is executable"
else
    echo "   ⚠️  WARNING: submit_train_improved.sh not executable"
    echo "      Fix: chmod +x submit_train_improved.sh"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -x submit_eval_improved.sh ]; then
    echo "   ✅ submit_eval_improved.sh is executable"
else
    echo "   ⚠️  WARNING: submit_eval_improved.sh not executable"
    echo "      Fix: chmod +x submit_eval_improved.sh"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 6: Required files exist
echo ""
echo "🔍 Checking required files..."
REQUIRED_FILES=(
    "config_improved.py"
    "train_improved.py"
    "evaluation_improved.py"
    "improvements_week2.py"
    "submit_train_improved.sh"
    "submit_eval_improved.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ ERROR: $file not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check 7: Dataset directories
echo ""
echo "🔍 Checking dataset directories..."
if [ -d "dataset/0" ] && [ -d "dataset/1" ] && [ -d "dataset/4" ]; then
    echo "   ✅ Datasets 0, 1, 4 found"
    COUNT_0=$(ls dataset/0/*.jpg 2>/dev/null | wc -l)
    COUNT_1=$(ls dataset/1/*.jpg 2>/dev/null | wc -l)
    COUNT_4=$(ls dataset/4/*.jpg 2>/dev/null | wc -l)
    echo "      Dataset 0: $COUNT_0 images"
    echo "      Dataset 1: $COUNT_1 images"
    echo "      Dataset 4: $COUNT_4 images"
else
    echo "   ❌ ERROR: Dataset directories not found"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "============================================================"
echo "Verification Summary"
echo "============================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo ""
    echo "✅ All checks passed!"
    echo ""
    echo "🚀 Ready to train:"
    echo "   ./submit_train_improved.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo ""
    echo "⚠️  $WARNINGS warning(s) - but OK to proceed"
    echo ""
    echo "🚀 Ready to train:"
    echo "   ./submit_train_improved.sh"
    echo ""
    exit 0
else
    echo ""
    echo "❌ $ERRORS error(s) found - please fix before proceeding"
    echo "   $WARNINGS warning(s)"
    echo ""
    exit 1
fi
