# V4 Advanced Preprocessing Pipeline

## Overview

V4 is the **final advanced preprocessing pipeline** for AIROGS glaucoma detection, integrating state-of-the-art preprocessing techniques with proven stable training methods.

## 🆕 New Features (V4)

### 1. **Optic Disk Detection & Cropping**
- Automatic detection via brightness thresholding (99th percentile)
- Intelligent cropping around optic disk (3.0x crop factor)
- Maintains 384x384 resolution after crop
- Fallback to original image if detection fails

### 2. **Advanced CLAHE (LAB Color Space)**
- Applies CLAHE in LAB color space (L channel only)
- Better contrast enhancement than RGB
- Preserves color information
- Grid size: 8x8, clip limit: 2.0

### 3. **Optional Vessel Enhancement**
- Frangi filter for blood vessel enhancement
- Useful for glaucoma-related vessel changes
- Can be toggled on/off via config

## ✅ Proven Methods (from V3)

- **Weighted Binary Cross-Entropy**: Stable loss function with 5:1 class weighting
- **Conservative Learning Rate**: 5e-5 (proven to work)
- **Moderate Augmentation**: Rotation (±15°), shifts (10%), horizontal flip, zoom (10%)
- **EfficientNet-B0 Backbone**: Lightweight, proven architecture
- **Callbacks**: Early stopping, ReduceLROnPlateau, model checkpointing

## 📂 Directory Structure

```
v4_advanced/
├── config_v4.py              # V4 configuration
├── dataset_v4.py             # V4 data generator with preprocessing
├── train_v4.py               # V4 training script
├── evaluate_v4.py            # V4 evaluation script
├── advanced_preprocessing.py # Preprocessing utilities
├── models/                   # Trained models
├── logs/                     # Training logs (CSV, JSON)
├── checkpoints/              # Model checkpoints
├── preprocessing_samples/    # Sample images showing preprocessing
└── evaluation_results_v4/    # Evaluation outputs
```

## 🚀 Quick Start

### 1. Train V4 Model

```bash
# Submit to SLURM cluster
sbatch submit_train_v4.sh

# Or run locally (not recommended, slow without GPU)
cd v4_advanced
python train_v4.py
```

### 2. Monitor Training

```bash
# Watch SLURM output
tail -f train_run/output/airogs_v4_*.out

# Check preprocessing samples after ~5 minutes
ls -lh v4_advanced/preprocessing_samples/
```

### 3. Evaluate Model

```bash
# Basic evaluation
python v4_advanced/evaluate_v4.py v4_advanced/models/airogs_v4_advanced_*_best.keras

# With Test-Time Augmentation (recommended)
python v4_advanced/evaluate_v4.py v4_advanced/models/airogs_v4_advanced_*_best.keras --tta
```

## 📊 Expected Results

### Baseline (No Advanced Preprocessing)
- **pAUC**: 0.4696
- **Sensitivity@95%**: 0.4833
- **AUC**: 0.8963

### V4 Goals
- **pAUC**: > 0.50 (target: +6% improvement)
- **Sensitivity@95%**: > 0.55 (target: +14% improvement)
- **AUC**: > 0.90 (maintain or improve)

## 🔧 Configuration Options

Edit `v4_advanced/config_v4.py`:

```python
# Preprocessing toggles
USE_OD_DETECTION = True          # Enable/disable optic disk detection
OD_CROP_FACTOR = 3.0             # Crop size multiplier (3.0 = 3x OD diameter)
USE_CLAHE = True                 # Enable/disable CLAHE
CLAHE_COLOR_SPACE = 'LAB'        # 'LAB', 'HSV', or 'RGB'
USE_VESSEL_ENHANCEMENT = False   # Enable/disable vessel enhancement

# Training parameters
LEARNING_RATE = 5e-5             # Proven stable LR
EPOCHS = 25                      # Longer training for preprocessing
BATCH_SIZE = 32                  # Fits in 24GB GPU
CLASS_WEIGHTS = {0: 1.0, 1: 5.0} # Address class imbalance
```

## 📈 Training Progress

V4 training takes **~10-12 hours** on RTX 6000:
- Preprocessing overhead: ~30% slower than V3
- Expected: 1350 steps/epoch × 25 epochs
- Checkpoints saved every epoch
- Early stopping if no improvement for 5 epochs

## 🔍 Debugging

### Check Preprocessing Samples
```bash
cd v4_advanced/preprocessing_samples
# View 10 sample images showing preprocessing steps
```

### Inspect Preprocessing Statistics
```bash
grep "Preprocessing Statistics" train_run/output/airogs_v4_*.out
```

### Verify OD Detection Rate
```bash
grep "Optic disk detected" train_run/output/airogs_v4_*.out | wc -l
# Should be > 80% of total images
```

## 🆚 Version Comparison

| Feature | Baseline | V2 (Failed) | V3 (Stable) | V4 (Advanced) |
|---------|----------|-------------|-------------|---------------|
| Loss | Weighted BCE | Focal Loss | Weighted BCE | Weighted BCE |
| LR | 5e-5 | 1e-4 | 5e-5 | 5e-5 |
| CLAHE | RGB | RGB | RGB | **LAB** |
| OD Detection | ❌ | ❌ | ❌ | **✅** |
| Vessel Enhancement | ❌ | ❌ | ❌ | **✅** |
| Epochs | 20 | 15 | 20 | 25 |
| pAUC | 0.4696 | 0.0776 | TBD | **TBD** |

## 📝 Notes

1. **Preprocessing Overhead**: V4 is ~30% slower due to OD detection and advanced CLAHE
2. **Memory Requirements**: 32GB RAM recommended (16GB minimum)
3. **GPU Requirements**: RTX 6000 (24GB VRAM) recommended
4. **TTA Evaluation**: Always use `--tta` for final evaluation (4x slower but more accurate)
5. **Sample Inspection**: Always check `preprocessing_samples/` to verify preprocessing quality

## 🐛 Troubleshooting

### Issue: "Out of memory" error
**Solution**: Reduce `BATCH_SIZE` in `config_v4.py` (try 16 or 8)

### Issue: OD detection failing (< 50% success rate)
**Solution**: Check image quality, adjust `OD_BRIGHTNESS_PERCENTILE` in `advanced_preprocessing.py`

### Issue: Training very slow
**Solution**: Disable vessel enhancement (`USE_VESSEL_ENHANCEMENT = False`)

### Issue: Poor performance despite preprocessing
**Solution**: 
1. Check preprocessing samples for quality
2. Try different `CLAHE_COLOR_SPACE` ('HSV' or 'RGB')
3. Adjust `OD_CROP_FACTOR` (try 2.5 or 3.5)

## 📚 References

- **Optic Disk Detection**: Brightness-based method (simple but effective)
- **CLAHE in LAB**: Pizer et al. "Adaptive Histogram Equalization and Its Variations" (1987)
- **Vessel Enhancement**: Frangi et al. "Multiscale vessel enhancement filtering" (1998)
- **EfficientNet**: Tan & Le "EfficientNet: Rethinking Model Scaling for CNNs" (2019)

## 🎯 Next Steps After V4 Training

1. **Evaluate with TTA**: `python v4_advanced/evaluate_v4.py MODEL_PATH --tta`
2. **Compare to Baseline**: Check if pAUC > 0.4696
3. **Analyze Failures**: Inspect misclassified images
4. **Iterate if Needed**: Adjust preprocessing parameters based on results
5. **Submit to Challenge**: Use best checkpoint for final submission

---

**Created**: 2024-12-17  
**Version**: 4.0 (Advanced Preprocessing)  
**Status**: Ready for training
