# PyEat

This repository contains an implementation of the EAT student-teacher model written in PyTorch and Lighting AI. The original code snippets come from the original EAT Repository and partially from the Fairseq Repository on GitHub. The current work resulted from transforming the main component, the Data2VecMultiModel intro a nn.Module class and building two seperate Lightning Modules for both pretraining and finetuning. In order to run this code, the following steps are required.

Create a virtual environment with
```conda create --name pyeat python=3.10```
```pip install -r requirements.txt```