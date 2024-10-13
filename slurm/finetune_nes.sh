#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=EAT-FINE
#SBATCH --output=/mnt/stud/work/phahn/repositories/EAT/logs/finetune_nes.log

source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate pyeat

cd /mnt/stud/work/phahn/repositories/EAT/PyEat/

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/EAT/output/finetune/NES/
MODEL_DIR=/mnt/stud/work/phahn/repositories/EAT/storage/
DATA_DIR=/mnt/stud/work/phahn/repositories/EAT/data/NES/
MODEL_PATH=/mnt/stud/work/phahn/repositories/EAT/storage/XCL/pretrained_weights_scaled_epoch_3.pth

echo "Saving results to $OUTPUT_DIR"
echo "Loading dataset from $DATA_DIR"
echo "Saving Model Weights to $MODEL_DIR"

srun python finetune.py \
    path.model_dir=$MODEL_DIR \
    path.output_dir=$OUTPUT_DIR \
    path.data_dir=$DATA_DIR \
    pretrained_weights_path=$MODEL_PATH \
    random_seed=42 \
    task=multilabel \
    finetune.n_epochs=50 \
    dataset.name=NES \
    dataset.num_classes=89 \
    finetune.load_pretrained_weights=True \