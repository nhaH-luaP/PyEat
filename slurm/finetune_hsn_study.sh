#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=EAT-FINE
#SBATCH --output=/mnt/stud/work/phahn/repositories/EAT/logs/finetune_hsn_study_%A_%a.log
#SBATCH --array=0-2%3

source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate pyeat

cd /mnt/stud/work/phahn/repositories/EAT/PyEat/

prediction_mode=(mean_pooling lin_softmax cls_token)
index=$SLURM_ARRAY_TASK_ID
pm=${prediction_mode[$index % 3]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/EAT/output/finetune/HSN/prediction_mode_study/$pm/
MODEL_DIR=/mnt/stud/work/phahn/repositories/EAT/storage/XCL/
DATA_DIR=/mnt/stud/work/phahn/repositories/EAT/data/HSN/
MODEL_PATH=/mnt/stud/work/phahn/repositories/EAT/storage/XCL/pretrained_weights_scaled_epoch_3.pth

echo "Saving results to $OUTPUT_DIR"
echo "Loading dataset from $DATA_DIR"
echo "Saving Model Weights to $MODEL_DIR"

srun python finetune.py \
    path.model_dir=$MODEL_DIR \
    path.output_dir=$OUTPUT_DIR \
    path.data_dir=$DATA_DIR \
    random_seed=42 \
    task=multilabel \
    finetune.n_epochs=25 \
    dataset.name=HSN \
    dataset.num_classes=21 \
    finetune.prediction_mode=$pm \
    finetune.load_pretrained_weights=True \
    finetune.train_linear_only=True \
    finetune.batch_size=64 \
    finetune.learning_rate=0.01 \
    pretrained_weights_path=$MODEL_PATH \
