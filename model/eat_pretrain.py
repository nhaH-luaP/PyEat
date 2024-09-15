import torch
import lightning as L


class EATPretrain(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, _ = batch['input_values'], batch['labels']
        
        # Forward Method of MultiModel
        result = self.model(x)

        #TODO: Which losses need to be extracted and optimized for pretraining EAT?
        loss = result["losses"]["cls"]

        # Logging
        self.log_dict({'train_loss': loss.item()})

        # Return Loss for optimization
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch['input_values'], batch['labels']
        
        # Forward Method of MultiModel
        result = self.model(x)

        loss = result["losses"]["cls"]

        # Logging
        self.log_dict({'val_loss': loss.item()})

        # Return Loss for optimization
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, weight_decay=5e-4, nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)
        return [optimizer], [lr_scheduler]