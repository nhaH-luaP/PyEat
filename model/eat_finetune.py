from .mixup import Mixup

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from sklearn.metrics import average_precision_score, roc_auc_score


class EATFineTune(L.LightningModule):
    def __init__(self, model, linear_classifier, num_classes, args, prediction_mode="mean_pooling"):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.prediction_mode = prediction_mode
        self.args = args
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
        val_metrics = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'val_loss': loss} | val_metrics)
    
    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_metrics = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'test_loss' : loss} | test_metrics)

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
    
    def calculate_metrics(self, logits, y, threshold=0.5):
        # Calculate Class Probabilities and Predictions
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= threshold).float()

        # Calculate Accuracy
        correct = (preds == y).float()
        test_acc = (correct.sum() / correct.numel()).item()

        # Calculate Hamming-Score
        y_int = y.int()
        preds_int = preds.int()
        ham_score = ((y_int & preds_int).sum(axis=1) / (y_int | preds_int).sum(axis=1)).mean().item()

        # Calculate AUROC
        try:
            auroc = roc_auc_score(y_true=y.cpu(), y_score=probas.cpu(), multi_class='ovr', average='weighted')
        except:
            auroc = 0.5

        # Calculate cmAP
        cmAP_micro = average_precision_score(y.cpu(), probas.cpu(), average="micro")
        cmAP_macro = average_precision_score(y.cpu(), probas.cpu(), average="macro")
        cmAP_weighted =  average_precision_score(y.cpu(), probas.cpu(), average="weighted")

        return {
            "test_acc" : test_acc, 
            "ham_score" : ham_score, 
            "auroc" : auroc, 
            "cmAP_micro" : cmAP_micro, 
            "cmAP_macro" : cmAP_macro, 
            "cmAP_weighted" : cmAP_weighted
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