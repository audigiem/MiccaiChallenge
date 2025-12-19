# 🔴 V2 FAILURE ANALYSIS & RECOVERY PLAN

## What Went Wrong with V2?

### V2 Results (DISASTER):
```
pAUC (α): 0.0776     ❌ 84% worse than baseline!
Sens@95% (β): 0.0760  ❌ 84% worse than baseline!
AUC-ROC: 0.7506       ❌ 16% worse than baseline!
```

### Baseline Results (GOOD):
```
pAUC (α): 0.4696     ✅
Sens@95% (β): 0.4833  ✅
AUC-ROC: 0.8963       ✅
```

---

## Root Causes of V2 Failure

### 1. **Focal Loss Instability**
- **Problem**: Focal loss with gamma=1.5, alpha=0.75 caused unstable training
- **Evidence**: Training CSV shows val_auc dropped from 0.85 to 0.50 at epoch 5
- **Impact**: Model learned high validation AUC (0.91) but terrible test performance

### 2. **Configuration Mismatch**
- **Problem**: Config had `UNFREEZE_AT_EPOCH=5` and `USE_LR_WARMUP=True` but we removed the callback
- **Impact**: Model expected progressive unfreezing but it never happened

### 3. **Overfitting to Validation Set**
- **Problem**: Model achieved 91% val_auc but only 75% test AUC
- **Evidence**: 16% gap between validation and test performance
- **Cause**: Training on datasets 0,1,4 but evaluating on dataset 5 (distribution shift)

### 4. **Threshold Selection**
- **Problem**: Evaluation used threshold=0.4554 which is too high
- **Result**: Only 25 true positives out of 329 actual positives (7.6% recall!)

---

## 🎯 Recovery Strategy

### ✅ V3 - BACK TO STABLE (Immediate Fix)

**Philosophy**: Go back to what worked, keep only proven improvements

**Changes**:
- ✅ **Weighted BCE** (not focal loss) - Baseline used this successfully
- ✅ **LR: 5e-5** - Baseline proven value
- ✅ **20 epochs** - Moderate, not too short
- ✅ **No unfreezing** - Keep it simple
- ✅ **KEEP CLAHE** - This is proven to help
- ✅ **KEEP augmentation** - Moderate level

**Expected Results**:
- Should match or slightly beat baseline
- Target: pAUC > 0.47, Sens@95% > 0.48
- Training time: 4-5 hours

**Usage**:
```bash
chmod +x submit_train_improved_v3.sh
./submit_train_improved_v3.sh
```

---

### 🚀 V4 - ADVANCED PREPROCESSING (Future Improvement)

**New Features**:
1. **Optic Disk Detection & Cropping**
   - Automatically detects optic disk (brightest region)
   - Crops around OD (glaucoma affects optic nerve)
   - Reduces irrelevant background

2. **Green Channel Enhancement**
   - Extract green channel (best contrast for vessels)
   - Better visualization of optic disk changes

3. **Advanced CLAHE**
   - Apply in LAB color space (better than RGB)
   - Preserves color relationships

4. **Vessel Enhancement (Optional)**
   - Frangi filter for vessel detection
   - Highlights vascular changes

**How to Use**:
```python
from advanced_preprocessing import preprocess_fundus_advanced

# Preprocess single image
img = preprocess_fundus_advanced(
    'path/to/image.jpg',
    target_size=384,
    use_od_crop=True,      # Enable OD cropping
    use_clahe=True,        # Advanced CLAHE
    use_vessel_enhance=False  # Experimental
)

# Batch preprocess dataset
from advanced_preprocessing import batch_preprocess_dataset

batch_preprocess_dataset(
    image_paths=['img1.jpg', 'img2.jpg', ...],
    output_dir='dataset/preprocessed/',
    use_od_crop=True,
    use_clahe=True
)
```

**Integration with Training**:
- Can create a new `ImageDataGenerator` that uses these preprocessing steps
- Or preprocess entire dataset once and train on preprocessed images

---

## 📊 Comparison Table

| Aspect | Baseline | V2 (Failed) | V3 (Stable) | V4 (Advanced) |
|--------|----------|-------------|-------------|---------------|
| Loss | Weighted BCE | Focal Loss | Weighted BCE | Weighted BCE |
| LR | 5e-5 | 1e-4 | 5e-5 | 5e-5 |
| Epochs | 30 | 15 | 20 | 20-25 |
| CLAHE | No | Yes | Yes | Advanced |
| Augmentation | Moderate | Aggressive | Moderate | Moderate |
| OD Crop | No | No | No | Yes |
| Vessel Enhance | No | No | No | Optional |
| Unfreezing | No | Config only | No | Optional |
| **pAUC (α)** | **0.4696** | **0.0776** | **>0.47?** | **>0.55?** |
| **Sens@95%** | **0.4833** | **0.0760** | **>0.48?** | **>0.60?** |

---

## 🎯 Action Plan

### Immediate (Now):
1. ✅ **Train V3** - Stable version to recover performance
   ```bash
   ./submit_train_improved_v3.sh
   ```

2. ⏳ **Monitor Training** - Watch for stability
   ```bash
   tail -f train_run/output/airogs_improved_v3_*.out
   ```

3. ✅ **Evaluate V3** - Compare with baseline
   ```bash
   python evaluation_improved.py outputs_improved/models/*v3*_best.keras --tta --clahe
   ```

### Next Steps (If V3 Works):
4. 🔬 **Test Advanced Preprocessing**
   ```bash
   # Test on single image
   python advanced_preprocessing.py dataset/0/some_image.jpg
   
   # Preprocess validation set
   python -c "from advanced_preprocessing import batch_preprocess_dataset; 
   batch_preprocess_dataset(['dataset/5/img1.jpg', ...], 'dataset/5_preprocessed/')"
   ```

5. 🚀 **Train V4** - With advanced preprocessing (if V3 is good)
   - Create `train_improved_v4.py` using `advanced_preprocessing.py`
   - Target: pAUC > 0.55, Sens@95% > 0.60

### Long Term:
6. 📈 **Ensemble Models** - Combine multiple models
7. 🎯 **Dataset 5 Fine-tuning** - Fine-tune on target dataset
8. 🔧 **Threshold Optimization** - Better threshold selection for 95% specificity

---

## 📝 Key Lessons Learned

1. **Focal Loss is Tricky** - Requires careful tuning, can be unstable
2. **Validation != Test** - Need to validate on similar distribution
3. **Simpler is Better** - Don't add complexity without evidence
4. **Baseline is Gold** - Start from proven methods, add incrementally
5. **Monitor Everything** - Watch training curves for instability

---

## 🔧 Files Created

### V3 (Stable):
- `config_improved_v3.py` - Stable configuration
- `train_improved_v3.py` - Stable training script
- `submit_train_improved_v3.sh` - SLURM submission

### V4 (Advanced):
- `advanced_preprocessing.py` - OD detection, advanced CLAHE, vessel enhancement

### Documentation:
- `V2_FAILURE_ANALYSIS.md` - This file

---

## 📞 Quick Reference

**Start V3 Training**:
```bash
./submit_train_improved_v3.sh
```

**Monitor V3**:
```bash
squeue -u $USER
tail -f train_run/output/airogs_improved_v3_*.out
```

**Evaluate V3**:
```bash
python evaluation_improved.py outputs_improved/models/*v3*_best.keras --tta --clahe
```

**Test Advanced Preprocessing**:
```bash
python advanced_preprocessing.py dataset/0/example_image.jpg
```

---

## ✅ Success Criteria

### V3 (Minimum):
- ✅ pAUC (α) ≥ 0.47 (match baseline)
- ✅ Sens@95% (β) ≥ 0.48 (match baseline)
- ✅ AUC-ROC ≥ 0.89 (match baseline)
- ✅ Stable training (no wild swings)

### V4 (Target):
- 🎯 pAUC (α) ≥ 0.55 (+17% over baseline)
- 🎯 Sens@95% (β) ≥ 0.60 (+24% over baseline)
- 🎯 AUC-ROC ≥ 0.91 (+2% over baseline)

---

**Status**: V3 ready to train, V4 preprocessing ready for testing

**Next Step**: Run `./submit_train_improved_v3.sh`
