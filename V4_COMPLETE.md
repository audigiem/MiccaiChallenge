# V4 ADVANCED PREPROCESSING PIPELINE - COMPLETE

## 🎉 Status: READY FOR TRAINING

Created: 2024-12-17  
Version: 4.0 (Advanced Preprocessing Integration)

---

## 📦 What's Been Created

### V4 Pipeline Structure
```
v4_advanced/
├── config_v4.py              ✅ V4 configuration with preprocessing flags
├── dataset_v4.py             ✅ Custom data generator with integrated preprocessing
├── train_v4.py               ✅ V4 training script with weighted BCE
├── evaluate_v4.py            ✅ V4 evaluation with TTA support
├── advanced_preprocessing.py ✅ OD detection, CLAHE, vessel enhancement
├── README.md                 ✅ Complete documentation
├── models/                   📁 For trained models
├── logs/                     📁 Training logs (CSV, JSON)
├── checkpoints/              📁 Model checkpoints
├── preprocessing_samples/    📁 Sample images
└── evaluation_results_v4/    📁 Evaluation outputs

Root directory:
├── submit_train_v4.sh        ✅ SLURM submission script (14 hours, 32GB, RTX 6000)
└── start_v4.sh               ✅ Interactive quick start helper
```

---

## 🆕 V4 Key Features

### 1. Advanced Preprocessing Integration
- **Optic Disk Detection**: Automatic detection via brightness thresholding
- **Intelligent Cropping**: 3.0x crop factor around detected OD
- **Advanced CLAHE**: LAB color space (L channel only) for better contrast
- **Vessel Enhancement**: Optional Frangi filter for blood vessels

### 2. Proven Training Methods (from V3)
- **Weighted BCE Loss**: Stable with 5:1 class weighting
- **Learning Rate**: 5e-5 (proven stable)
- **Moderate Augmentation**: Rotation, shifts, flip, zoom
- **EfficientNet-B0**: Lightweight backbone
- **Smart Callbacks**: Early stopping, LR reduction, checkpointing

### 3. Production-Ready Features
- **Preprocessing Statistics**: Track OD detection success rate
- **Sample Visualization**: Auto-save preprocessing examples
- **TTA Evaluation**: 4 augmentations for robust predictions
- **Comprehensive Logging**: JSON history, CSV logs, config snapshots

---

## 🚀 How to Use

### Option 1: Interactive Quick Start (Recommended)
```bash
cd /user/8/audigiem/FIB/DLMA/MiccaiChallenge/MiccaiChallenge
./start_v4.sh
```
This will:
- ✅ Check all V4 files exist
- ✅ Validate Python syntax
- ✅ Verify datasets are present
- ✅ Optionally submit training job

### Option 2: Direct Submission
```bash
cd /user/8/audigiem/FIB/DLMA/MiccaiChallenge/MiccaiChallenge
sbatch submit_train_v4.sh
```

### Monitor Training
```bash
# Watch live output
tail -f train_run/output/airogs_v4_*.out

# Check job status
squeue -u $USER

# View preprocessing samples (after ~5 minutes)
ls -lh v4_advanced/preprocessing_samples/
```

### Evaluate Model
```bash
# Basic evaluation
python v4_advanced/evaluate_v4.py v4_advanced/models/*_best.keras

# With TTA (recommended for final results)
python v4_advanced/evaluate_v4.py v4_advanced/models/*_best.keras --tta
```

---

## 📊 Expected Training Timeline

| Time | Event |
|------|-------|
| 0-5 min | Environment setup, data loading |
| 5 min | First batch processed, preprocessing samples saved |
| 10 min | Epoch 1 complete (~30 min/epoch with preprocessing) |
| 5 hours | ~10 epochs complete |
| 10-12 hours | Training complete (early stopping likely) |

**Total Time**: 10-12 hours on RTX 6000

---

## 🎯 Performance Goals

### Baseline (No Advanced Preprocessing)
- **pAUC**: 0.4696
- **Sensitivity@95%**: 0.4833
- **AUC**: 0.8963

### V4 Target Improvements
- **pAUC**: > 0.50 (+6% minimum)
- **Sensitivity@95%**: > 0.55 (+14% minimum)
- **AUC**: > 0.90 (maintain or improve)

### Success Criteria
✅ **PASS**: pAUC > 0.47 (better than baseline)  
🎯 **GOOD**: pAUC > 0.50 (meaningful improvement)  
⭐ **EXCELLENT**: pAUC > 0.55 (significant improvement)

---

## 🔧 Configuration Highlights

### Preprocessing (`config_v4.py`)
```python
USE_OD_DETECTION = True          # Optic disk detection & cropping
OD_CROP_FACTOR = 3.0             # 3x optic disk diameter
USE_CLAHE = True                 # Advanced CLAHE
CLAHE_COLOR_SPACE = 'LAB'        # LAB color space (best contrast)
USE_VESSEL_ENHANCEMENT = False   # Vessels (optional, slower)
```

### Training Parameters
```python
LEARNING_RATE = 5e-5             # Proven stable
EPOCHS = 25                      # Longer for preprocessing benefits
BATCH_SIZE = 32                  # Fits RTX 6000 24GB VRAM
CLASS_WEIGHTS = {0: 1.0, 1: 5.0} # Address 8:1 imbalance
```

### Augmentation
```python
AUGMENTATION = {
    'rotation_range': 15,        # ±15° rotation
    'width_shift_range': 0.1,    # 10% horizontal shift
    'height_shift_range': 0.1,   # 10% vertical shift
    'horizontal_flip': True,     # Mirror flip
    'zoom_range': 0.1,           # 10% zoom
    'fill_mode': 'constant',     # Black padding
    'cval': 0
}
```

---

## 📈 What Happens During Training

### 1. Initialization (0-5 min)
- GPU setup & memory allocation
- Dataset loading (~43,000 training images)
- Model compilation with weighted BCE
- Preprocessing statistics initialization

### 2. First Epoch (5-35 min)
- **~1350 steps** (43,200 images / 32 batch size)
- Preprocessing applied to each batch:
  1. Load image
  2. Detect optic disk (if enabled)
  3. Crop around OD
  4. Apply advanced CLAHE (LAB)
  5. Apply augmentation
  6. Normalize (0-1)
- Save 10 preprocessing samples
- Validate on ~400 steps

### 3. Subsequent Epochs (30 min each)
- Continue training with preprocessing
- Track best validation AUC
- Reduce LR if plateau (factor=0.5, patience=3)
- Early stop if no improvement (patience=5)

### 4. Completion (10-12 hours)
- Save best checkpoint
- Save final model
- Export training history (CSV + JSON)
- Print preprocessing statistics

---

## 🔍 Debugging & Verification

### Check Preprocessing Quality
```bash
cd v4_advanced/preprocessing_samples
# View 10 sample images showing:
# - Original
# - OD detection
# - Cropped region
# - CLAHE enhanced
```

### Inspect Preprocessing Stats
```bash
grep "Preprocessing Statistics" train_run/output/airogs_v4_*.out
```
Expected output:
```
Preprocessing Statistics:
  Total images processed: 43200
  Optic disk detected: 38880 (90.0%)
  Detection failures: 4320 (10.0%)
  Average detection time: 0.05s
```

### Verify Training Progress
```bash
tail -20 train_run/output/airogs_v4_*.out
```
Look for:
- ✅ Accuracy increasing (0.60 → 0.85+)
- ✅ Val AUC increasing (0.70 → 0.90+)
- ✅ Loss decreasing (0.5 → 0.2)
- ❌ Val AUC dropping (overfitting - early stop will trigger)

---

## 🆚 Version Comparison

| Feature | Baseline | V2 (Failed) | V3 (Stable) | **V4 (Advanced)** |
|---------|----------|-------------|-------------|-------------------|
| Loss | Weighted BCE | Focal | Weighted BCE | **Weighted BCE** |
| LR | 5e-5 | 1e-4 | 5e-5 | **5e-5** |
| CLAHE | RGB | RGB | RGB | **LAB** ⭐ |
| OD Detection | ❌ | ❌ | ❌ | **✅** ⭐ |
| OD Cropping | ❌ | ❌ | ❌ | **✅** ⭐ |
| Vessels | ❌ | ❌ | ❌ | **✅** (opt) ⭐ |
| Epochs | 20 | 15 | 20 | **25** |
| Training Time | 8h | 6h | 8h | **10-12h** |
| pAUC | 0.4696 | 0.0776 💥 | TBD | **TBD** 🎯 |

---

## 📝 Important Notes

### DO's ✅
- ✅ Always check `preprocessing_samples/` after first epoch
- ✅ Use `--tta` flag for final evaluation
- ✅ Monitor OD detection success rate (should be > 80%)
- ✅ Compare to baseline (0.4696 pAUC) before celebrating
- ✅ Keep V3 model as backup (proven stable)

### DON'Ts ❌
- ❌ Don't panic if first epoch is slow (~30 min is normal)
- ❌ Don't modify config during training
- ❌ Don't use vessel enhancement unless needed (adds 40% overhead)
- ❌ Don't evaluate without TTA (less reliable)
- ❌ Don't delete baseline/V3 models until V4 proves better

### Troubleshooting

**Problem**: Training very slow (>40 min/epoch)  
**Solution**: Disable vessel enhancement or reduce batch size

**Problem**: OD detection rate < 50%  
**Solution**: Check image quality, adjust `OD_BRIGHTNESS_PERCENTILE`

**Problem**: Out of memory error  
**Solution**: Reduce `BATCH_SIZE` to 16 or 8

**Problem**: Poor preprocessing quality  
**Solution**: Try different `CLAHE_COLOR_SPACE` ('HSV' or 'RGB')

---

## 🎓 Technical Details

### Optic Disk Detection Algorithm
1. Convert to grayscale
2. Find 99th percentile brightness (likely OD)
3. Threshold at 90% of max brightness
4. Morphological operations (opening + closing)
5. Find largest connected component
6. Fit ellipse to contour
7. Validate diameter (20-300 pixels)

### Advanced CLAHE Process
1. Convert RGB → LAB color space
2. Extract L (lightness) channel
3. Apply CLAHE (grid=8x8, clip=2.0)
4. Replace L channel with enhanced version
5. Convert LAB → RGB
6. Maintain original color information

### Vessel Enhancement (Optional)
1. Apply Frangi filter (multi-scale)
2. Detect tubular structures (vessels)
3. Blend with original image (alpha=0.3)
4. Enhance vessel visibility for analysis

---

## 📚 File Reference

### `config_v4.py` (170 lines)
V4 configuration with all preprocessing flags, training parameters, paths

### `dataset_v4.py` (304 lines)
Custom `V4AdvancedImageDataGenerator` class integrating preprocessing into Keras pipeline

### `train_v4.py` (240 lines)
Complete training script with GPU setup, data loading, model compilation, training loop

### `evaluate_v4.py` (195 lines)
Evaluation script with TTA support, metrics computation, results export

### `advanced_preprocessing.py` (468 lines)
Core preprocessing utilities: OD detection, CLAHE, vessel enhancement

### `submit_train_v4.sh` (70 lines)
SLURM submission script: 14h, 32GB, RTX 6000, 8 CPUs

### `start_v4.sh` (100 lines)
Interactive helper: checks prerequisites, validates files, submits job

---

## 🎯 Next Steps

### 1. Start Training (NOW)
```bash
cd /user/8/audigiem/FIB/DLMA/MiccaiChallenge/MiccaiChallenge
./start_v4.sh
```

### 2. Monitor First Epoch (After 30 min)
```bash
# Check training is progressing
tail -f train_run/output/airogs_v4_*.out

# Verify preprocessing samples
ls -lh v4_advanced/preprocessing_samples/
```

### 3. Wait for Completion (10-12 hours)
```bash
# Check if training finished
grep "TRAINING COMPLETED" train_run/output/airogs_v4_*.out
```

### 4. Evaluate with TTA
```bash
# Get best model path
best_model=$(ls -t v4_advanced/models/*_best.keras | head -1)

# Evaluate with TTA
python v4_advanced/evaluate_v4.py $best_model --tta
```

### 5. Compare Results
```bash
# V4 results
cat v4_advanced/evaluation_results_v4/*/evaluation_report.txt

# Baseline for comparison
cat evaluation_results_b0DA/evaluation_results.txt
```

### 6. Decision Time
- **If V4 > Baseline**: 🎉 Success! Use V4 for submission
- **If V4 ≈ Baseline**: 🤔 Analyze preprocessing samples, try adjustments
- **If V4 < Baseline**: 🔧 Disable OD detection or try different CLAHE color space

---

## 📞 Support

For issues, check:
1. `v4_advanced/README.md` - Detailed documentation
2. `train_run/stderr/airogs_v4_*.err` - Error logs
3. Preprocessing samples - Visual quality check
4. Training history CSV - Metric trends

---

## ✅ Final Checklist

- [x] V4 folder structure created
- [x] Configuration file (`config_v4.py`)
- [x] Custom data generator (`dataset_v4.py`)
- [x] Training script (`train_v4.py`)
- [x] Evaluation script (`evaluate_v4.py`)
- [x] Preprocessing utilities (`advanced_preprocessing.py`)
- [x] SLURM submission script (`submit_train_v4.sh`)
- [x] Quick start helper (`start_v4.sh`)
- [x] Documentation (`README.md`)
- [x] All files syntax-validated
- [x] Scripts made executable

**Status**: ✅ **READY FOR TRAINING**

---

**Good luck with V4! 🚀**

The advanced preprocessing should provide meaningful improvements over the baseline. Remember to check the preprocessing samples and compare results carefully.
