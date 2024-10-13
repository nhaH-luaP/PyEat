# PyEat

This repository contains an implementation of the EAT student-teacher model written in PyTorch and Lighting AI specifically for images. It contains code snippets from both the [EAT-Repository](https://github.com/cwx-worst-one/EAT/tree/main) and the [Fairseq-Repository](https://github.com/facebookresearch/fairseq/tree/main) but eliminates the usual workflow of Fairseq-Models for a better readability of code.

## Usage

Create a virtual environment with
```conda create --name pyeat python=3.10```
Clone the repository [BirdSet](https://github.com/DBD-research-group/BirdSet/tree/main) and run ```pip install -e .``` to install.
In addition, the following adaptions are necessary:
  - ```pip install timm```
  - ```pip install numpy==1.26.4```

## Features
We provide the following Features inside this repo:
 - Pretraining a Data2Vec-Model on images with variable size.
 - Finetuning on Downstream-Tasks for linear evaluation.

In general, the last line in each file in the [slurm](./slurm)-folder provide examples of how to use these two files and examples for arguments to pass.

Decisive Hyperparameters:
 - All Hyperparameters of the shape ```path.*``` controll settings concerning where to save/load models and data from.
 - All Hyperparameters of the shape ```pretrain.*``` controll settings concerning the pretraining like hyperparameters for optimizer and lr_scheduler.
 - All Hyperparameters of the shape ```finetune.*```, again, controll settings concerning the finetuning like hyperparameters for optimizer and lr_scheduler. In addition, we provide settings such as ```load_pretrained_weights``` to use a pretrained model, ```prediction_mode``` for different ways to generate Features out of the Transformer-Encoder or ```class_weighted_loss``` to account for strongly imbalanced datasets such as HSN.

## Current Challenges

 - Most runs on a compute cluster (except HSN and XCL) wont progress when preparing the data using the birdset repository. Main assumption why this is the case is the slowness of the cluster itself.
 - A lot of studies are required to debug the repository in a greater setting. A few of these are listed below:
  - Investigate different prediction modes
  - Investigate the usage of mixup for improved finetuning
  - Investigate different checkpoints of pretraining to see if training longer helps
  - Investigate different methods for finetuning such as linear evaluation vs full model finetuning on downstream task or training multiclass vs. multilabel setting.
