import pytorch_lightning
import pytorchvideo.models
import torch.nn.functional as F
import torch

class SlowfastModule(pytorch_lightning.LightningModule):
    def __init__(self, args=None):
        self.args = args
        super().__init__()

        self.model = pytorchvideo.models.create_slowfast(
            input_channels=(3,3),
            model_num_class=6,
        )
        self.batch_key = 'video'
    
    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        # import pdb; pdb.set_trace()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        # import pdb; pdb.set_trace()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=1e-5,
            momentum=0.9,
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 1, last_epoch=-1
        )
        return [optimizer], [scheduler]