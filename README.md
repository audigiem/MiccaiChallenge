# AIROGS Glaucoma Detection - Complete Implementation

A comprehensive solution for the AIROGS challenge on automated detection of referable glaucoma from color fundus photographs. This project demonstrates systematic development through baseline implementation, iterative data-centric improvements, and extensive evaluation achieving challenge-competitive performance.

## Project Structure

```
.
├── config.py                 # Baseline configuration
├── config_improved.py        # V1 configuration (focal loss)
├── config_improved_v3.py     # V3 configuration (best base model)
├── config_v4.py              # V4 configuration (advanced preprocessing)
├── config_v5_final.py        # V5 configuration (optimized)
├── dataset.py                # Dataset loading and preprocessing
├── model.py                  # Model architecture and compilation
├── evaluation.py             # Evaluation metrics and visualization
├── train.py                  # Main training script
├── inference.py              # Inference script with TTA support
├── train_cluster_baseline.sh # SLURM script for baseline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── 0/, 1/, 4/            # Training images from datasets 0, 1, 4
│   └── train_labels.csv      # Training labels
├── evaluation_result_*/      # Evaluation results for each model version
│   ├── evaluation_results.txt
│   ├── training.csv
│   ├── history.json
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── prediction_distribution.png
└── report/                   # Comprehensive analysis
    ├── AIROGS_Report.ipynb   # Full report with visualizations
    ├── generate_report.py    # Automated figure generation
    └── plots/                # Generated comparison figures
```

## Results Summary

### Performance Overview

Five model versions were developed and evaluated on 11,442 test images:

| Model | pAUC | Sensitivity @ 95% | AUC-ROC | F1-Score | Description |
|-------|------|-------------------|---------|----------|-------------|
| Baseline | 0.6246 | 0.6778 | 0.9225 | 0.3076 | Weighted BCE, basic augmentation |
| Baseline + TTA | 0.6919 | 0.7508 | 0.9434 | 0.3503 | +10.8% with test-time augmentation |
| V1 (Focal Loss) | 0.5987 | 0.6565 | 0.9220 | 0.2878 | Failed experiment: unstable training |
| V3 (Enhanced DA) | **0.6767** | **0.7416** | 0.9362 | 0.3383 | Best base model: CLAHE + moderate aug |
| V3 + TTA | **0.7626** | **0.8663** | **0.9586** | **0.4286** | Best overall: exceeds 0.75 target |
| V4 (Advanced) | 0.5826 | 0.6383 | 0.9124 | 0.2712 | Failed: aggressive cropping lost context |
| V4 + TTA | 0.6816 | 0.7538 | 0.9445 | 0.3569 | Largest TTA gain (+17.0%) |
| V5 (Final) | 0.6154 | 0.6809 | 0.9287 | 0.3096 | Optimized V4, still below V3 |
| V5 + TTA | 0.6800 | 0.7538 | 0.9464 | 0.3603 | Recovered but not optimal |

**Key Achievement:** V3+TTA reached pAUC 0.7626 and sensitivity 86.63%, exceeding the challenge target of 0.75 sensitivity at 95% specificity.

### Model Evolution

**Baseline (Week 1)**
- EfficientNet-B0 with weighted BCE (1:5 class weights)
- Basic augmentation (flips, rotation +/-15 degrees)
- Learning rate: 5e-5
- Result: Solid foundation, pAUC 0.6246

**V1: Focal Loss Experiment (Week 2)**
- Hypothesis: Focal loss better handles 1:29 class imbalance
- Result: FAILED - Training instability, pAUC dropped to 0.5987 (-4.1%)
- Lesson: Weighted BCE provides more stable training for medical imaging

**V3: Best Base Model (Week 2)**
- Strategy: Return to weighted BCE, keep beneficial improvements
- Key changes: CLAHE preprocessing, moderate augmentation, 1:8 class weights, LR 7.5e-5
- Result: SUCCESS - pAUC 0.6767 (+8.3% vs baseline)
- Impact: CLAHE preprocessing proved most valuable single improvement

**V4: Advanced Preprocessing (Week 3)**
- Hypothesis: Optic disc detection and targeted cropping improves detection
- Changes: Circular Hough transform for disc detection, 3x disc radius ROI, LAB color space CLAHE
- Result: FAILED - pAUC 0.5826 (-13.9% vs V3)
- Lesson: Aggressive cropping lost peripheral retinal context; full-image approaches superior

**V5: Optimized Configuration (Week 3)**
- Strategy: Keep V4 preprocessing, optimize hyperparameters
- Changes: Higher LR (1e-4), increased class weights (1:10), longer patience (7 epochs)
- Result: Improved over V4 (pAUC 0.6154) but still below V3
- Lesson: Hyperparameter tuning cannot overcome fundamental architectural limitations

**Test-Time Augmentation Impact**
- Applied 5-augmentation ensemble (center + 4 corners with flips)
- Improvements: +10.8% (Baseline), +12.7% (V3), +17.0% (V4), +10.5% (V5)
- V3+TTA achieved best overall performance: pAUC 0.7626, sensitivity 86.63%

## Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 384×384 RGB fundus images
- **Output**: Binary classification (RG vs NRG)
- **Classification head**: 3 dense layers (256-128-1) with dropout (0.3, 0.2)

### Training Configuration
- **Hardware**: Dell R740 cluster with NVIDIA RTX 6000 GPUs (24GB VRAM)
- **Framework**: TensorFlow 2.x / Keras
- **Batch size**: 32
- **Optimizer**: Adam with ReduceLROnPlateau scheduler
- **Training time**: Approximately 2 hours per model (20 epochs)
- **Dataset size**: Approximately 54,000 training images from datasets 0, 1, and 4

### Key Techniques

**CLAHE Preprocessing (V3+)**
- Contrast Limited Adaptive Histogram Equalization
- Parameters: clip limit 2.0, tile grid 8x8
- Impact: +8.3% pAUC improvement over baseline
- Benefit: Enhanced optic disc and vessel visibility

**Class Imbalance Handling**
- Challenge: 1:29 RG to NRG ratio (329 vs 11,113 test cases)
- Solution: Weighted Binary Cross-Entropy
- Optimal weights: 1:8 for V3 (vs 1:5 baseline, 1:10 for V5)

**Data Augmentation**
- Horizontal/vertical flips
- Rotation: +/-15 degrees
- Zoom: +/-10%
- Brightness: 0.8-1.2 range
- Strategy: Moderate augmentation outperformed aggressive preprocessing

**Test-Time Augmentation**
- 5-augmentation ensemble: center crop + 4 corners
- Horizontal flip applied to each
- Prediction averaging across augmentations
- Consistent 10-17% performance improvement

### Evaluation Metrics

**Challenge-Specific Metrics**
- **Partial AUC (pAUC)**: Area under ROC curve for 90-100% specificity (target: >0.70)
- **Sensitivity @ 95% Specificity**: True positive rate at fixed high specificity (target: >0.75)

**Standard Metrics**
- **AUC-ROC**: Overall area under receiver operating characteristic curve
- **Accuracy**: Overall correct classification rate
- **Precision**: Positive predictive value (TP / (TP + FP))
- **Recall**: Sensitivity (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate (TN / (TN + FP))

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your data as follows:
```
data/
├── 0/, 1/, 4/            # Folders containing training images from datasets
│   ├── TRAIN000000.jpg
│   ├── TRAIN000001.jpg
│   └── ...
└── train_labels.csv      # CSV with columns: challenge_id, class
```

### 3. Train Models

**Baseline Model**
```bash
python train.py --config config.py
```

**Best Model (V3)**
```bash
python train.py --config config_improved_v3.py
```

**Train on Cluster with SLURM**
```bash
# Edit SLURM script for your cluster configuration
sbatch train_cluster_baseline.sh

# Check job status
squeue -u $USER

# View output
tail -f logs/airogs_*.out
```

### 4. Run Evaluation

```bash
# Evaluate single model
python evaluation.py \
    --model path/to/model.keras \
    --test-dir data/test \
    --test-labels data/test_labels.csv \
    --output-dir evaluation_results/

# Evaluation with TTA
python inference.py \
    --model path/to/model.keras \
    --test-dir data/test \
    --test-labels data/test_labels.csv \
    --use-tta \
    --output evaluation_tta.json
```

### 5. Generate Report

```bash
cd report/

# Generate all comparison figures
python generate_report.py

# View comprehensive analysis
jupyter notebook AIROGS_Report.ipynb
```
## Key Findings

### What Worked

1. **CLAHE Preprocessing**: Single most impactful improvement (+8.3% pAUC)
2. **Weighted BCE Loss**: More stable than focal loss for medical imaging
3. **Moderate Augmentation**: Balanced generalization without overfitting
4. **Test-Time Augmentation**: Consistent +10-17% improvement across all models
5. **Systematic Development**: Incremental changes with careful evaluation

### What Failed

1. **Focal Loss (V1)**: Training instability, erratic validation metrics
2. **Aggressive Cropping (V4)**: Lost peripheral retinal context (RNFL, vessels)
3. **Complex Preprocessing**: LAB color space added complexity without benefit
4. **Over-engineering**: Domain knowledge did not always translate to performance gains

### Lessons Learned

1. **Start simple, add complexity incrementally**: V3's straightforward approach outperformed complex V4/V5
2. **Data-centric improvements matter most**: Preprocessing and augmentation provided larger gains than algorithmic changes
3. **Validate domain assumptions**: Optic disc-focused cropping seemed logical but reduced performance
4. **Metric alignment is critical**: Optimizing for standard AUC does not necessarily optimize partial AUC
5. **Test-time augmentation is powerful**: Should be standard practice for production systems

### Clinical Implications

**Strengths**
- High specificity (>95%) minimizes unnecessary referrals
- 86.6% sensitivity (V3+TTA) detects majority of glaucoma cases
- Fast inference enables high-throughput screening

**Limitations**
- 13.4% false negative rate (V3+TTA) means some cases missed
- Performance on early-stage or atypical presentations unknown
- Limited to specific imaging protocols and populations

**Deployment Recommendations**
- Best suited as first-line triage tool with expert oversight
- Use prediction confidence scores to guide referral decisions
- Regular monitoring and model recalibration needed
- Consider ensemble approaches for production systems

## Configuration Files

The project includes multiple configuration files for different model versions:

**config.py** - Baseline
```python
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 5e-5
MODEL_BACKBONE = "efficientnet-b0"
CLASS_WEIGHTS = {0: 1.0, 1: 5.0}  # Addressing 1:29 imbalance
```

**config_improved_v3.py** - Best Base Model
```python
IMAGE_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 7.5e-5  # Increased from baseline
MODEL_BACKBONE = "efficientnet-b0"
CLASS_WEIGHTS = {0: 1.0, 1: 8.0}  # Increased minority weight
USE_CLAHE = True  # Key improvement
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
```

**config_v4.py** - Advanced (Failed Experiment)
```python
USE_OPTIC_DISC_DETECTION = True
CROP_TO_DISC_ROI = True
ROI_RADIUS_MULTIPLIER = 3.0
USE_LAB_COLOR_SPACE = True
CLASS_WEIGHTS = {0: 1.0, 1: 5.0}
```

**config_v5_final.py** - Optimized
```python
LEARNING_RATE = 1e-4  # Higher than baseline
CLASS_WEIGHTS = {0: 1.0, 1: 10.0}  # Maximum minority weight tested
EARLY_STOPPING_PATIENCE = 7  # More patient
```

## Comprehensive Report

A detailed analysis report is available in the `report/` directory:

**AIROGS_Report.ipynb** - Jupyter notebook containing:
- Complete methodology description
- Model evolution and development strategy
- Performance comparison across all versions
- Training dynamics analysis
- Confusion matrix analysis
- ROC curves and prediction distributions
- Clinical implications and deployment considerations

**generate_report.py** - Automated report generation:
- Loads results from all model evaluation folders
- Generates comparison tables (CSV)
- Creates performance evolution plots
- Produces confusion analysis visualizations
- Generates separate training curve plots (loss, AUC, precision, recall)
- Copies and organizes individual model figures

To view the complete report:
```bash
cd report/
jupyter notebook AIROGS_Report.ipynb
```

## Repository Structure Details

**Training Scripts**
- `train.py` - Main training script with configurable parameters
- `train_cluster_baseline.sh` - SLURM script for cluster execution
- `inference.py` - Inference with TTA support

**Core Modules**
- `dataset.py` - Data loading, preprocessing, augmentation pipeline
- `model.py` - Model architecture, compilation, callbacks
- `evaluation.py` - Metrics computation, visualization generation
- `utils.py` - Utility functions for file operations

**Configuration Files**
- `config.py` - Baseline configuration
- `config_improved.py` - V1 configuration (focal loss)
- `config_improved_v3.py` - V3 configuration (best model)
- `config_v4.py` - V4 configuration (optic disc detection)
- `config_v5_final.py` - V5 configuration (optimized)

**Evaluation Outputs**
- `evaluation_result_b0_FullDS/` - Baseline results
- `evaluation_improveBaselien_v1/` - V1 results
- `evaluation_boDA_v3/` - V3 results
- `evaluation_boDA_v4/` - V4 results
- `evaluation_boDA_v5/` - V5 results

Each evaluation folder contains:
- `evaluation_results.txt` - Detailed metrics
- `training.csv` - Training history
- `history.json` - Training history (JSON format)
- `roc_curve.png` - ROC curve visualization
- `confusion_matrix.png` - Confusion matrix
- `prediction_distribution.png` - Prediction histogram
- Model configuration JSON (for V3, V4, V5)

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in config file
BATCH_SIZE = 16  # or 8

# Reduce image size
IMAGE_SIZE = 256
```

### Slow Training
```python
# Enable mixed precision (enabled by default)
USE_MIXED_PRECISION = True

# Reduce augmentation complexity
# Edit dataset.py augmentation pipeline
```

### Poor Performance
```python
# Check class weights are appropriate
CLASS_WEIGHTS = {0: 1.0, 1: 8.0}  # V3 optimal

# Ensure CLAHE is enabled (V3+)
USE_CLAHE = True

# Verify data loading correctly
# Check dataset.py preprocessing pipeline
```

### SLURM Job Issues
```bash
# Check job status
squeue -u $USER

# View detailed job info
scontrol show job <job_id>

# Check output logs
cat logs/airogs_*.out

# Verify GPU allocation
nvidia-smi
```

## 📚 References

1. AIROGS Challenge: https://airogs.grand-challenge.org/
2. Challenge Paper: https://doi.org/10.1167/iovs.63.8.3
3. EfficientNet: https://arxiv.org/abs/1905.11946
4. Medical Image Analysis: Best practices for fundus image analysis

## 📄 License

This is an educational project for the DLMA course.

## 👥 Author

Matteo - FIB/UPC DLMA Course

## 🙏 Acknowledgments

- AIROGS challenge organizers
- TensorFlow/Keras community
- Course instructors

