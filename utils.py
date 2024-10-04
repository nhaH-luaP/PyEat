import random
import os
import numpy as np
import torch
from lightning import Callback
import logging
import time
import datetime
import logging
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from collections import defaultdict, deque

import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


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




class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )



class MetricLogger(L.Callback):
    def __init__(self, log_interval=20, delimiter=' ', use_print=False):
        super().__init__()
        self.log_interval = log_interval
        self.delimiter = delimiter
        self.use_print = use_print

        self.logger = logging.getLogger(__name__)
        self.header = f"Epoch [{0}]"
        self.meters = defaultdict(SmoothedValue)

    def _log(self, log_msg):
        if self.use_print:
            print(log_msg)
        else:
            self.logger.info(log_msg)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def on_train_start(self, trainer, pl_module) -> None:
        self._start_time_train = time.time()

    def on_train_end(self, trainer, pl_module) -> None:
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train))
        log_msg = self.delimiter.join([
            f'{self.header} Total training time: {eta}',
        ])
        self._log(log_msg)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time_train_epoch = time.time()

        self.train_step = 0
        self.header = f"Epoch [{trainer.current_epoch}]"
        self.num_batches = len(trainer.train_dataloader)
        self.space_fmt = f":{len(str(self.num_batches))}d"

        self.meters = defaultdict(SmoothedValue)
        self.meters['lr'] = SmoothedValue(window_size=1, fmt="{value:.4f}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        metrics['lr'] = trainer.optimizers[0].param_groups[0]['lr']

        for key, val in metrics.items():
            batch_size = len(batch['input_values'])
            self.meters[key].update(val, n=batch_size)

        if self.train_step % self.log_interval == 0:
            log_msg = self.delimiter.join([
                f'{self.header}',
                ("[{0" + self.space_fmt + "}").format(batch_idx)+f"/{self.num_batches}]",
                str(self),
            ])
            self._log(log_msg)
        self.train_step += 1

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train_epoch))
        log_msg = f"{self.header} Total time: {eta}"
        self._log(log_msg)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.header = f"Epoch [{trainer.current_epoch}]"
        self._start_time_val = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_val))
        log_msg = self.delimiter.join([
            f'{self.header} Total time for validation: {eta}',
        ])
        self._log(log_msg)
