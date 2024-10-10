from model.data2vecmultimodel import Data2VecMultiModel
from model.eat_finetune import EATFineTune
from utils import seed_everything, MetricsCallback, MetricLogger

import hydra
import os
import logging
import json
import torch
from omegaconf import OmegaConf

import lightning as L

from birdset.datamodule.base_datamodule import DatasetConfig, LoadersConfig, BirdSetTransformsWrapper
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.configs.datamodule_configs import LoaderConfig

from torchvision import transforms
from tqdm import tqdm

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Print Args for Identification
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directories necessary for output and model savings
    os.makedirs(args.path.output_dir, exist_ok=True)

    # Enable Reproducability
    seed_everything(args.random_seed)

    # Build birdset datamodule
    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir=args.path.data_dir,
            hf_path=args.path.hf_path,
            hf_name=args.dataset.name,
            n_workers=args.dataset.num_workers,
            val_split=args.dataset.val_split,
            task=args.task,
            classlimit=500,
            eventlimit=5,
            sampling_rate=32000,
        ),
        loaders=LoadersConfig(train=LoaderConfig(batch_size=args.finetune.batch_size, shuffle=True, drop_last=True),
                              valid=LoaderConfig(batch_size=64, shuffle=False),
                              test=LoaderConfig(batch_size=64, shuffle=False)),
        transforms=BirdSetTransformsWrapper(
            task=args.task,
            #spectrogram_augmentations={'normalize':transforms.Normalize(mean=10.3461275100708, std=6.643364906311035)},
            )
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    # Calculate Label-Imbalance
    # label_distribution = torch.zeros(size=(args.dataset.num_classes,))
    # for batch in tqdm(dm.train_dataloader()):
    #     y = batch['labels']
    #     label_distribution += torch.sum(y, dim=0)
    

    if args.dataset.name == 'HSN':
        label_distribution = torch.tensor([56.,  971.,  218.,  360.,  400.,  285., 1229.,  478.,  322.,   80., 417., 1221., 1076., 1699.,  536.,  375.,  405.,  531.,  669., 1469., 643.])
    elif args.dataset.name == 'NBP':
        label_distribution = torch.tensor([1287.,  953.,  748., 1199., 1265., 1423., 1304., 1057., 1421.,  948., 1330.,  982., 1164., 1340., 1248., 1377., 1051., 1249., 1263., 1196., 
                                       795., 1284., 1299., 1409.,  907., 1252.,  952.,  549., 1547., 1437., 1027., 1099., 1018.,  547., 1318., 1350., 1077., 1019., 1293.,  841., 1204., 
                                       1285.,  710., 1074.,  888.,  719., 1120., 1217.,  904., 1090., 1244.])
    else:
        label_distribution = torch.ones(size=(args.dataset.num_classes,))
    
    label_weights = 1/label_distribution
    logging.info(f"Label Distribution is as follows: {label_distribution}")

    # Initialize Model with potentially pretrained weights
    logging.info(f">>> Initialize Model.")
    backbone = Data2VecMultiModel(args=args)
    if args.finetune.load_pretrained_weights and os.path.exists(args.pretrained_weights_path):
        logging.info(f"Loading Pretrained Weights from {args.pretrained_weights_path}")
        backbone.load_state_dict(torch.load(args.pretrained_weights_path))
    linear_classifier = torch.nn.Linear(in_features=args.multimodel.embed_dim, out_features=args.dataset.num_classes)
    model = EATFineTune(model=backbone, linear_classifier=linear_classifier, num_classes=args.dataset.num_classes, args=args, label_weights=label_weights)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()
    metrics_logger = MetricLogger()

    # Finetune Model
    trainer = L.Trainer(
        max_epochs=args.finetune.n_epochs, 
        callbacks=[metrics_callback, metrics_logger],
        default_root_dir=args.path.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator='gpu'
        )
    trainer.fit(model=model, datamodule=dm)

    # Evaluate Model
    dm.setup(stage='test')
    trainer.test(model=model, datamodule=dm)

    # Extract metrics and export into json file
    metrics_dict = {'train_metrics':metrics_callback.train_metrics, 'test_metrics':metrics_callback.test_metrics}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(metrics_dict, f)


if __name__ == '__main__':
    main()   
