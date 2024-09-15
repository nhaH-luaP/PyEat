# PyEat

This repository contains an implementation of the EAT student-teacher model written in PyTorch and Lighting AI. The original code snippets come from the original EAT Repository and partially from the Fairseq Repository on GitHub. The current work resulted from transforming the main component, the Data2VecMultiModel intro a nn.Module class and building two seperate Lightning Modules for both pretraining and finetuning. It is currently NOT running and requires probably a lot more debugging and cleaning. Here is some information that may help when working on this repository:
- I do not yet understand all the args sitting in the config. There may be some settings that resolve some issues.
- Also, there are some args doubled in multimodel and modality subargs, so there may be some issues there.
- There is a requirement.txt that contains all the packages i installed so far but i have not tested if it can be installed without issues.
- The files inside the folder model stem mainly from the EAT-Repository while the files from the subfolder fairseq stem from the fairseq-Repository to exclude its import as it clashes with the hydra config manager.
- I currently use the BirdSet Data Module with the HSN dataset but it may be easier to build a CIFAR10 datamodule for debugging.
- Current Bugs mainly resolve arround GPU errors such as out-of-memory or problems with some tensors being on cuda while others are on cpu. May be resolved when running on the server but not quite sure.
- There is currently no implementation of saving the weights in pretrain and loading the weights in finetune. This should be easily implemented by reducing it to the weights of the Data2VecMultiModel and not saving a checkpoint of the lightning modules.
- These are some args that I've used for debugging to reduce the memory on CUDA:

```
"multimodel.average_top_k_layers=1",
"multimodel.depth=1",
"multimodel.num_heads=1",
"modality.prenet_depth=1",
"modality.num_alibi_heads=1",
"modality.model_depth=1",
"modality.embed_dim=32",
"multimodel.embed_dim=32"
```