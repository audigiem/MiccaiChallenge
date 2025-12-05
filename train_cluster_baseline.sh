#!/bin/bash

# Enhanced cluster execution script with detached execution
# Usage: ./training_augmentedUNet.sh [--time=20:00:00] [--mem=8GB] [--cpus=12]

# Default values
MEMORY="4GB"
CPUS="8"
TIME_LIMIT="4:00:00"
SCRIPT_NAME="train.py"
ENV_PATH="~/MiccaiChallenge/bin/activate"
JOB_NAME="baseline_training"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mem=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --cpus=*)
            CPUS="${1#*=}"
            shift
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--time=20:00:00] [--mem=8GB] [--cpus=12]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p run/{stderr,output,logs,checkpoints}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="run/slurm_job_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=run/output/${JOB_NAME}_${TIMESTAMP}.out
#SBATCH --error=run/stderr/${JOB_NAME}_${TIMESTAMP}.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

# Log job start
echo "[$(date '+%Y-%m-%d %H:%M:%S')]  Job started on node: \$SLURM_NODELIST"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ID: \$SLURM_JOB_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration: MEM=${MEMORY}, CPUS=${CPUS}, TIME=${TIME_LIMIT}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU: \$CUDA_VISIBLE_DEVICES"

# Load CUDA modules (adjust versions if needed)
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "No CUDA module found, using system CUDA"
module load cudnn/8.9 2>/dev/null || module load cudnn 2>/dev/null || echo "No cuDNN module found, using system cuDNN"

# Display loaded modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
module list

# Activate environment and run training
source ${ENV_PATH}
export OMP_NUM_THREADS=${CPUS}
export TF_CPP_MIN_LOG_LEVEL=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Python training script..."
python3 ${SCRIPT_NAME}
PYTHON_EXIT_CODE=\$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed with exit code: \$PYTHON_EXIT_CODE"

# Final summary
if [ \$PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully!"

    # Show final metrics if available
    if grep -q "Mean.*dice.*coefficient" run/output/${JOB_NAME}_${TIMESTAMP}.out 2>/dev/null; then
        final_dice=\$(grep "Mean.*dice.*coefficient" run/output/${JOB_NAME}_${TIMESTAMP}.out | tail -1)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final result: \$final_dice"
    fi

    # List saved models
    models=\$(find . -name "*.keras" -newer "$SLURM_SCRIPT" 2>/dev/null)
    if [ ! -z "\$models" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]  Models saved:"
        echo "\$models" | while read model; do
            size=\$(ls -lh "\$model" | awk '{print \$5}')
            echo "   \$model (\$size)"
        done
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training failed!"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job completed"
EOF

# Submit the job
echo "Submitting training job to SLURM..."
echo "Configuration: MEM=$MEMORY, CPUS=$CPUS, TIME=$TIME_LIMIT"

JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Output: run/output/${JOB_NAME}_${TIMESTAMP}.out"
    echo "Errors: run/stderr/${JOB_NAME}_${TIMESTAMP}.err"
    echo ""
    echo "Monitor your job with:"
    echo "   squeue -u audigiem                                    # Check job status"
    echo "   tail -f run/output/${JOB_NAME}_${TIMESTAMP}.out       # Follow output"
    echo "   tail -f run/stderr/${JOB_NAME}_${TIMESTAMP}.err       # Follow errors"
    echo "   scancel $JOB_ID                                       # Cancel if needed"
    echo ""
    echo "You can now safely close this terminal - the job will continue running!"
else
    echo "Failed to submit job"
    exit 1
fi
