#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=EAT-PRE
#SBATCH --output=/mnt/stud/work/phahn/repositories/EAT/logs/finetune_%x.log

source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate pyeat

cd /mnt/stud/work/phahn/repositories/EAT/PyEat/

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/PyEat/output/
MODEL_DIR=/mnt/stud/work/phahn/repositories/PyEat/storage/
DATA_DIR=/mnt/stud/work/phahn/datasets/

echo "Saving results to $OUTPUT_DIR"
echo "Loading dataset from $DATA_DIR"
echo "Saving Model Weights to $MODEL_DIR"

srun python finetune.py \
    path.model_dir=$MODEL_DIR \
    path.output_dir=$OUTPUT_DIR \
    path.data_dir=$DATA_DIR \
    random_seed=42 \
    task=multilabel \
