from model.eat import Data2VecMultiModel
import torch
import hydra
import os
import logging
import json
from birdset.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
import random
import numpy as np
from omegaconf import OmegaConf
from lightning import Callback
import lightning as L


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

    dm = build_dataset(args)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize Model
    logging.info(f">>> Initialize Model.")
    model = Data2VecMultiModel(args=args)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()

    # Finetune Model
    trainer = L.Trainer(max_epochs=args.model.n_epochs, callbacks=[metrics_callback], accelerator='gpu')
    trainer.fit(model=model, datamodule=dm)

    # Evaluate Model
    dm.setup(stage='test')
    trainer.test(model=model, datamodule=dm)

    # Extract metrics and export into json file
    metrics_dict = {'train_metrics':metrics_callback.train_metrics, 'test_metrics':metrics_callback.test_metrics}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(metrics_dict, f)





from birdset.configs.datamodule_configs import LoaderConfig

def build_dataset(args):
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
        loaders=LoadersConfig(train=LoaderConfig(batch_size=1))
    )
    return dm



def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})

    def on_val_epoch_end(self, trainer, pl_module):
        self.val_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})

    def on_test_epoch_end(self, trainer, pl_module):
        self.test_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})


if __name__ == '__main__':
    main()   