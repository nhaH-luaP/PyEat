from .mixup import Mixup
from .utils import TopKAccuracy

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from torchmetrics.classification import MultilabelAUROC, Accuracy
from torchmetrics.classification.average_precision import MultilabelAveragePrecision

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class EATFineTune(L.LightningModule):
    def __init__(self, model, linear_classifier, num_classes, args, device='cuda'):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.args = args
        
        self.prediction_mode = args.finetune.prediction_mode
        self.train_linear_only = args.finetune.train_linear_only
        self.num_classes = num_classes
        
        self.loss_type = args.finetune.loss_type
        self.regularize_negatives = args.finetune.regularize_negatives
        self.sigma = args.finetune.sigma

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
        self.use_mixup = args.finetune.use_mixup
        
        self.accuracy_fn = TopKAccuracy()
        self.auroc_fn = MultilabelAUROC(num_labels=num_classes)
        self.cmap_fn = MultilabelAveragePrecision(num_labels=num_classes)


    def training_step(self, batch, batch_idx):
        x, y = batch['input_values'], batch['labels']

        # For mixup, shape divisible by two is important due to mixup workflow to have an evenly shaped batch
        if x.shape[0] % 2 == 0 and self.args.finetune.use_mixup: 
            x, y = self.mixup_fn(x, y)
        
        # Get logits and collect gradients depending on if you only want to finetune last layer or more.
        if self.train_linear_only:
            with torch.no_grad():
                result = self.model(x, features_only=True, remove_extra_tokens=(self.prediction_mode == "cls_token"), mask=False)
        else:
            result = self.model(x, features_only=True, remove_extra_tokens=(self.prediction_mode == "cls_token"), mask=False)
        features = result['x']
        reduced_features = self.reduce_features(features)
        logits = self.linear_classifier(reduced_features)

        # Calculate Loss differently depending on the chosen loss_type (multiclass or multilabel)
        if self.loss_type == 'multilabel' or self.use_mixup:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none' if self.regularize_negatives else 'mean')
            #TODO: Experimental. Trying to reduce the impact of negative class instances on loss.
            if self.regularize_negatives:
                weight = y + self.sigma
                loss = loss * weight
                loss = loss.mean()
        elif self.loss_type == 'multiclass':
            loss = nn.functional.cross_entropy(logits, y)
        else:
            raise AssertionError('Loss Type not found!')

        # Metrics
        train_metrics = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'train_loss' : loss.item()} | {'train_'+key : value for key, value in train_metrics.items()})

        # Return Loss for optimization
        return loss


    def validation_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        with torch.no_grad():
            result = self.model(x, features_only=True, remove_extra_tokens=(self.prediction_mode == "cls_token"), mask=False)
            features = result['x']
            reduced_features = self.reduce_features(features)
            logits = self.linear_classifier(reduced_features)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_metrics = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({'test_loss' : loss} | {'test_'+key : value for key, value in test_metrics.items()})


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.linear_classifier.parameters() if self.train_linear_only else self.parameters(), 
            lr=self.args.finetune.learning_rate, 
            weight_decay=self.args.finetune.weight_decay, 
            nesterov=self.args.finetune.nesterov, 
            momentum=self.args.finetune.momentum
            )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.finetune.n_epochs)
        return [optimizer], [lr_scheduler]
    

    def calculate_metrics(self, logits, y):
        probas = torch.nn.functional.sigmoid(logits)
        acc = self.accuracy_fn(probas, y).item()
        cmAP = self.cmap_fn(logits, y.long()).item()
        auroc = self.auroc_fn(probas, y.long()).item()

        return {
            'top1': 0 if (acc != acc) else acc, 
            'cmAP' : 0 if (cmAP != cmAP) else cmAP, 
            'AUROC' : 0 if (auroc != auroc) else auroc
        }


    def reduce_features(self, features):
        # Incoming Features have the shape (Batch Size, Number of Patches + Extra Tokens, Embedding Size)
        # This feature reduction method reduces the features coming from the Data2Vec model according to different schemes
        # Mean Pooling Averages over all 
        if self.prediction_mode == "mean_pooling": # mean_pooling averages over all patches
            features = features.mean(dim=1)
        elif self.prediction_mode == "cls_token": # cls_token only takes the class token, which is the first patch
            features = features[:, 0]
        elif self.prediction_mode == "lin_softmax": # lin softmax is a bit more complex and not considered in this work 
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