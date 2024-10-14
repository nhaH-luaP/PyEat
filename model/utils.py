import torch
import torchmetrics


class TopKAccuracy(torchmetrics.Metric):
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, targets):
        # Get the top-k predictions
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        #consider no_call instances (a threshold is needed here!)
        if self.include_nocalls:
            #check if top-k predictions for all-negative instances are less than threshold 
            no_positive_predictions = preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            correct_all_negative = (no_call_targets & no_positive_predictions.all(dim=1))

        else:
            #no_calls are removed, set to 0
            correct_all_negative = torch.tensor(0).to(targets.device)

        #convert one-hot encoded targets to class indices for positive cases
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)
        
        #update correct and total, excluding all-negative instances if specified
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total