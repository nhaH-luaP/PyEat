from model.eat_pretrain import EATPretrain
from model.data2vecmultimodel import Data2VecMultiModel
from utils import seed_everything, MetricsCallback, MetricLogger

import hydra
import os
import torch
import logging
import json
from omegaconf import OmegaConf

import lightning as L

from birdset.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.configs.datamodule_configs import LoaderConfig


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Print Args for Identification
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directories necessary for output and model savings
    logging.info(f"Best model will be saved in {args.path.model_dir} !")
    os.makedirs(args.path.model_dir, exist_ok=True)
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
        loaders=LoadersConfig(train=LoaderConfig(batch_size=args.pretrain.batch_size))
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize Model
    logging.info(f">>> Initialize Model.")
    backbone = Data2VecMultiModel(args=args)
    n_steps = (dm.len_trainset//dm.train_batch_size) * args.pretrain.n_epochs
    logging.info(f"Total Number of Updates will be {n_steps}!")
    model = EATPretrain(model=backbone, args=args, n_steps=n_steps)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()
    metrics_logger = MetricLogger(log_interval=args.pretrain.log_interval)

    # Pretrain the Model
    trainer = L.Trainer(
        max_epochs=args.pretrain.n_epochs, 
        callbacks=[metrics_callback, metrics_logger],
        default_root_dir=args.path.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator='gpu'
        )
    trainer.fit(model=model, datamodule=dm)

    # Extract the final backbone and save the state dict
    backbone = model.model
    state_dict = backbone.state_dict()
    torch.save(state_dict, os.path.join(args.path.model_dir, "pretrained_weights_"+str(args.random_seed)+"final.pth"))

    # Extract metrics and export into json file
    metrics_dict = {'train_metrics':metrics_callback.train_metrics, 'test_metrics':metrics_callback.test_metrics}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(metrics_dict, f)


if __name__ == '__main__':
    main()   
