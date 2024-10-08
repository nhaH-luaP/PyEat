import os
import torch
import logging
import lightning as L


class EATPretrain(L.LightningModule):
    def __init__(self, model, args, n_steps):
        super().__init__()
        self.model = model
        self.args = args
        self.n_steps = n_steps

    def training_step(self, batch, batch_idx):
        x, _ = batch['input_values'], batch['labels']
        
        # Forward Method of MultiModel
        result = self.model(x)

        #TODO: Which losses need to be extracted and optimized for pretraining EAT?
        cls_loss = result["losses"]["cls"].mean()
        d2v_loss = result["losses"]["d2v"].mean() * self.args.pretrain.d2v_scale
        loss = cls_loss + d2v_loss

        # Logging
        self.log_dict({'train_total_loss': loss.item(), 'train_cls_loss': cls_loss.item(), 'train_d2v_loss': d2v_loss.item()})

        # Return Loss for optimization
        return loss
    
    def on_train_epoch_end(self):
        # Save a model after each epoch to see how pretraining-length influences model performance on downstream task
        path = os.path.join(self.args.path.model_dir, "pretrained_weights_"+self.args.pretrain.name_suffix+"_epoch_"+str(self.current_epoch+1)+".pth")
        logging.info(f"Saving model after Epoch {self.current_epoch} to {path}!")
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch['input_values'], batch['labels']
        
        # Forward Method of MultiModel
        result = self.model(x)

        cls_loss = result["losses"]["cls"].mean()
        d2v_loss = result["losses"]["d2v"].mean() * self.args.pretrain.d2v_scale
        loss = cls_loss + d2v_loss

        # Logging
        self.log_dict({'val_total_loss': loss.item(), 'val_cls_loss': cls_loss.item(), 'val_d2v_loss': d2v_loss.item()})

        # Return Loss for optimization
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.parameters(), 
            lr=self.args.pretrain.learning_rate, 
            weight_decay=self.args.pretrain.weight_decay, 
            nesterov=self.args.pretrain.nesterov, 
            momentum=self.args.pretrain.momentum
            )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.n_steps)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": None,
        }
        return [optimizer], [lr_scheduler_config]