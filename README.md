# PyEat

This repository contains an implementation of the EAT student-teacher model written in PyTorch and Lighting AI specifically for spectogram images. It contains code snippets from both the [EAT-Repository](https://github.com/cwx-worst-one/EAT/tree/main) and the [Fairseq-Repository](https://github.com/facebookresearch/fairseq/tree/main) but eliminates the usual workflow of Fairseq-Models for a better readability of code.

## Usage

Create a virtual environment with
```conda create --name pyeat python=3.10```
Clone the repository [BirdSet](https://github.com/DBD-research-group/BirdSet/tree/main) and run ```pip install -e .``` to install.
In addition, the following adaptions are necessary:
  - ```pip install timm```
  - ```pip install numpy==1.26.4```

## Current Stand of Things

Currently, this repository is still in development and tests are running on a cluster. The results and potential debugging are available afterwards, which is approximatly 14 Days for pretraining and another two days for finetuning.
