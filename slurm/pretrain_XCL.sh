#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=EAT-PRE
#SBATCH --output=/mnt/stud/work/phahn/repositories/EAT/logs/pretrain_XCL.log

source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate pyeat

cd /mnt/stud/work/phahn/repositories/EAT/PyEat/

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/EAT/output/
MODEL_DIR=/mnt/stud/work/phahn/repositories/EAT/storage/XCL/
DATA_DIR=/mnt/stud/work/phahn/repositories/EAT/data2/XCL/

echo "Saving results to $OUTPUT_DIR"
echo "Loading dataset from $DATA_DIR"
echo "Saving Model Weights to $MODEL_DIR"

srun python pretrain.py \
    path.model_dir=$MODEL_DIR \
    path.output_dir=$OUTPUT_DIR \
    path.data_dir=$DATA_DIR \
    random_seed=42 \
    dataset.name=XCL \
    dataset.num_classes=9735 \
    task=multiclass \
    pretrain.batch_size=16 \
    multimodel.clone_batch=6 \
    pretrain.n_epochs=5 \
