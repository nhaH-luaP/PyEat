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

## Current Stand of Things

Currently, this repository is still in development and tests are running on a cluster. The results and potential debugging are available afterwards, which is approximatly 14 Days for pretraining and another two days for finetuning.
