from model.data2vecmultimodel import Data2VecMultiModel
from model.eat_finetune import EATFineTune
from utils import seed_everything, MetricsCallback

import hydra
import os
import logging
import json
import torch
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
            dataset_name=args.dataset.name,
            hf_path=args.path.hf_path,
            hf_name=args.dataset.name,
            n_workers=args.dataset.num_workers,
            val_split=args.dataset.val_split,
            task=args.task,
            classlimit=500,
            eventlimit=5,
            sampling_rate=32000,
        ),
        loaders=LoadersConfig(train=LoaderConfig(batch_size=16))
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize Model
    logging.info(f">>> Initialize Model.")
    backbone = Data2VecMultiModel(args=args)
    linear_classifier = torch.nn.Linear(in_features=args.multimodel.embed_dim, out_features=args.dataset.num_classes)
    model = EATFineTune(model=backbone, linear_classifier=linear_classifier, num_classes=args.dataset.num_classes)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()

    # Finetune Model
    trainer = L.Trainer(max_epochs=args.model.n_epochs, callbacks=[metrics_callback])
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