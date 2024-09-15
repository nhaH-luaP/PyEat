from .mixup import Mixup

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from sklearn.metrics import average_precision_score, accuracy_score


class EATFineTune(L.LightningModule):
    def __init__(self, model, linear_classifier, num_classes, prediction_mode="mean_pooling", optim_params={"weight_decay":5e-4, "learning_rate":1e-1, "n_epochs":1}):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.prediction_mode = prediction_mode
        self.optim_params = optim_params
        self.mixup_fn = Mixup(
                mixup_alpha=0.5,
                cutmix_alpha=0.5,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.0,
                mode="batch",
                label_smoothing=0.0,
                num_classes=num_classes,
            )

    def training_step(self, batch, batch_idx):
        # Perform Mixup and then get the logits
        x, y = batch['input_values'], batch['labels']
        if x.shape[0] % 2 == 0: # Important due to mixup workflow to have an evenly shaped batch
            x, y = self.mixup_fn(x, y)
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Logging
        self.log_dict({'train_loss': loss.item()})

        # Return Loss for optimization
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, mAP = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'val_loss': loss, 'val_acc': test_acc, 'val_mAP':mAP})
    
    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, mAP = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'test_loss': loss, 'test_acc': test_acc, 'mAP':mAP})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_params["learning_rate"], weight_decay=self.optim_params["weight_decay"], nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.optim_params["n_epochs"])
        return [optimizer], [lr_scheduler]
    
    def get_logits(self, x):
        with torch.no_grad():
            result = self.model(x, features_only=True, remove_extra_tokens=True, mask=False) #TODO: What about the remove extra tokens option?
        features = result['x']

        # Different prediction modes to work with the features resulting from the fairseq model. Shape: (Batch-Size, Dim1, Dim2)
        features = self.reduce_features(features)

        logits = self.linear_classifier(features)
        return logits
    
    def calculate_metrics(self, logits, y):
        # Calculate Accuracy in a multi-label setting
        probas = torch.nn.functional.sigmoid(logits)
        preds = probas.flatten() >= 0.5
        test_acc = accuracy_score(y_true=y.flatten().cpu(), y_pred=preds.cpu())
        mAP = average_precision_score(y_true=y.cpu(), y_score=probas.cpu())
        return test_acc, mAP

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