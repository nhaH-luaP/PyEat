from .mixup import Mixup
from .utils import TopKAccuracy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification.average_precision import MultilabelAveragePrecision

from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class EATFineTune(L.LightningModule):
    def __init__(self, model, linear_classifier, num_classes, args):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.args = args
        self.prediction_mode = self.args.finetune.prediction_mode
        self.mixup_fn = Mixup(
                mixup_alpha=args.finetune.mixup_alpha,
                cutmix_alpha=args.finetune.cutmix_alpha,
                cutmix_minmax=None,
                prob=args.finetune.mix_prob,
                switch_prob=0.0,
                mode="batch",
                label_smoothing=args.finetune.mixup_label_smoothing,
                num_classes=num_classes,
            )

        self.accuracy_fn = TopKAccuracy(topk=1, threshold=args.finetune.threshold)
        self.auroc_fn = MultilabelAUROC(num_labels=num_classes)
        self.cmap_fn = MultilabelAveragePrecision(num_labels=num_classes, threshold=None, average="macro")

    def training_step(self, batch, batch_idx):
        # Get logits of either normal or mixed samples
        # Shape divisible by two is important due to mixup workflow to have an evenly shaped batch
        x, y = batch['input_values'], batch['labels']
        if x.shape[0] % 2 == 0 and self.args.finetune.use_mixup: 
            x, y = self.mixup_fn(x, y)
        logits = self.get_logits(x)

        # Calculate Loss
        if self.args.finetune.loss_type == 'multilabel' or self.args.finetune.use_mixup:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)

        # Logging
        self.log_dict({'train_loss': loss.item()})

        # Return Loss for optimization
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        if self.args.finetune.loss_type == 'multilabel':
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)

        acc = self._calculate_accuracy(logits, y.argmax(-1))

        self.log_dict({'val_loss': loss, 'val_acc':acc})

    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_metrics = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'test_loss' : loss} | {'test_'+key : value for key, value in test_metrics.items()})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.linear_classifier.parameters(), 
            lr=self.args.finetune.learning_rate, 
            weight_decay=self.args.finetune.weight_decay, 
            nesterov=self.args.finetune.nesterov, 
            momentum=self.args.finetune.momentum
            )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.finetune.n_epochs)
        return [optimizer], [lr_scheduler]
    
    def get_logits(self, x):
        with torch.no_grad():
            result = self.model(x, features_only=True, remove_extra_tokens=True, mask=False)
        features = result['x']

        # Different prediction modes to work with the features resulting from the fairseq model. Shape: (Batch-Size, Dim1, Dim2)
        features = self.reduce_features(features)

        logits = self.linear_classifier(features)
        return logits
    
    def _calculate_accuracy(self, logits, labels):
        probas = logits.softmax(-1)
        preds = probas.argmax(-1)
        n_samples = preds.size(0)
        n_correct = torch.sum(preds == labels).item()
        acc = 0 if n_samples == 0 else n_correct/n_samples
        return acc
    
    def _calculate_mAP(self, output, target):
        classes_num = target.shape[-1]
        ap_values = {}
        for k in range(classes_num):
            avg_precision = average_precision_score(target[:, k], output[:, k], average=None)
            ap_values[k] = avg_precision
        mean_ap = np.nanmean(list(ap_values.values()))
        return mean_ap, ap_values
    
    def _calculate_hamming_score(self, y_true, y_pred):
        return (
            (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)
        ).mean()
    
    def calculate_metrics(self, logits, y):
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).cpu().numpy().astype(int)

        acc = self.accuracy_fn(probas, y)
        #ham = self._calculate_hamming_score(y_pred=preds, y_true=y.cpu().numpy().astype(int))
        mAP, _ = self._calculate_mAP(target=y.cpu(), output=probas.cpu())
        cmAP = self.cmap_fn(logits, y.long())
        auroc = self.auroc_fn(probas, y.long())
        return {
            'accuracy': acc, 
         #   'hamming_score' : ham, 
            'mAP' : mAP, 
            'cmAP' : cmAP, 
            'AUROC' : auroc
        }

    def reduce_features(self, features):
        if self.prediction_mode == "mean_pooling":
            features = features.mean(dim=1)
        elif self.prediction_mode == "cls_token":
            features = features[:, 0]
        elif self.prediction_mode == "lin_softmax":
            dtype = features.dtype
            features = F.logsigmoid(features.float())
            features = torch.logsumexp(features + features, dim=1) - torch.logsumexp(features + 1e-6, dim=1)
            features = features.clamp(max=0)
            features = features - torch.log(-(torch.expm1(features)))
            features = torch.nan_to_num(features, nan=0, posinf=0, neginf=0)
            features = features.to(dtype=dtype)
        else:
            raise Exception(f"unknown prediction mode {self.prediction_mode}")
        return features
