import logging
import re
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import torch
import torch.nn.functional as F
from torchvision.models.video.mvit import mvit_v2_s
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from Tools.mvit import build_mvit_v2_b
from coral_pytorch.losses import corn_loss, coral_loss
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch, proba_to_label
from slowfast.utils.parser import load_config
from slowfast.models import build_model
from argparse import Namespace

"""
GRASSPClassificationModule

This LightningModule implementation constructs the desired model architecture,
configures the training/validation loop, calculates metrics, and configures 
the optimizer. 
"""

class GRASSPClassificationModule(pytorch_lightning.LightningModule):
    def __init__(self, args=None):
        self.args = args
        super().__init__()

        # Instantiate metrics functions
        self.train_accuracy = MulticlassAccuracy(num_classes=self.args.num_classes+1)
        self.train_MAE = MeanAbsoluteError()
        self.val_accuracy = MulticlassAccuracy(num_classes=self.args.num_classes+1)
        self.val_MAE = MeanAbsoluteError()

        self.sanity_check = False
        self.val_preds = []
        self.val_target = []
        self.val_loss = []
        self.val_tasks = {}

        # Construct desired model architecture
        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=self.args.num_classes,
            )
            self.batch_key = "video"
        elif self.args.arch == 'slowfast':
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                input_channels=(3,3),
                model_num_class=self.args.num_classes,
                slowfast_channel_reduction_ratio=int(1/self.args.slowfast_beta),
                slowfast_conv_channel_fusion_ratio=self.args.slowfast_fusion_conv_channel_ratio,
                slowfast_fusion_conv_kernel_size=(self.args.slowfast_fusion_kernel_size, 1, 1),
                slowfast_fusion_conv_stride=(self.args.slowfast_alpha, 1, 1), 
            )
            self.batch_key = "video"
            ## Transfer Learning: freeze all layers except final one(s)
            if self.args.transfer_learning:
                # Load pre-trained state dict
                state_dict = torch.load(self.args.pretrained_state_dict)
                # Remove final layer from state_dict for compatibility with different sized final layers
                state_dict.pop("blocks.6.proj.weight")
                state_dict.pop("blocks.6.proj.bias")
                self.model.load_state_dict(state_dict, strict=False)
                # Save pointers to layers to unfreeze
                block4_pathway0_res_block2 = self.model.blocks[4].multipathway_blocks[0].res_blocks[2]
                block4_pathway1_res_block2 = self.model.blocks[4].multipathway_blocks[1].res_blocks[2]
                if not self.args.finetune:
                    # Freeze params
                    for param in self.model.parameters():
                        param.requires_grad = False
                    # Unfreeze saved layers
                    for param in block4_pathway0_res_block2.parameters(): param.requires_grad = True
                    for param in block4_pathway1_res_block2.parameters(): param.requires_grad = True
                # Construct last fc layer
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = torch.nn.Linear(num_features, self.args.num_classes)
            
            ## If using CORN Ordinal regression, replace final layer with -1 output nodes.
            if self.args.ordinal and self.args.ordinal_strat == 'CORN':
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = torch.nn.Linear(num_features, self.args.num_classes-1)
            ## If using CORAL Ordinal regression, replace final layer with a CORAL weight-sharing layer.
            if self.args.ordinal and self.args.ordinal_strat == 'CORAL':
                num_features = self.model.blocks[6].proj.in_features
                self.model.blocks[6].proj = CoralLayer(size_in=num_features, num_classes=self.args.num_classes)

        elif self.args.arch == 'mvit_v2_b':
            self.model = build_mvit_v2_b(
                spatial_size = self.args.video_crop_size,
                temporal_size = self.args.num_frames,
            )
            if self.args.transfer_learning:
                # Load state_dict
                state_dict = torch.load(self.args.pretrained_state_dict)
                self.model.load_state_dict(state_dict, strict=True)
                # Save pointers to layers to unfreeze
                last_block_mlp  = self.model.blocks[-1].mlp
                last_block_sd   = self.model.blocks[-1].stochastic_depth
                model_norm      = self.model.norm
                model_head      = self.model.head
                if not self.args.finetune:
                    # Freeze layers
                    for param in self.model.parameters():
                        param.requires_grad = False
                    # Unfreeze saved last layers
                    for param in last_block_mlp.parameters(): param.requires_grad = True
                    for param in last_block_sd.parameters(): param.requires_grad = True
                    for param in model_norm.parameters(): param.requires_grad = True
                    for param in model_head.parameters(): param.requires_grad = True
            # Construct last FC layer
            num_features = self.model.head[1].in_features
            self.model.head[1] = torch.nn.Linear(num_features, self.args.num_classes)

            ## If using CORN Ordinal regression, replace final layer with -1 output nodes.
            if self.args.ordinal and self.args.ordinal_strat == 'CORN':
                num_features = self.model.head[1].in_features
                self.model.head[1] = torch.nn.Linear(num_features, self.args.num_classes-1)
            if self.args.ordinal and self.args.ordinal_strat == 'CORAL':
                num_features = self.model.head[1].in_features
                self.model.head[1] = CoralLayer(size_in=num_features, num_classes=self.args.num_classes)

            self.batch_key = "video"
        elif self.args.arch == 'mvit_maskfeat':
            pyslowfast_args = Namespace(opts=None)
            pyslowfast_cfg = load_config(pyslowfast_args, args.pyslowfast_cfg_file)
            self.model = build_model(pyslowfast_cfg)
            if self.args.transfer_learning:
                # Load state_dict
                state_dict = torch.load(self.args.pretrained_state_dict)['model_state']
                self.model.load_state_dict(state_dict, strict=False)
                # Disable final activation layer (done outside in loss calculation)
                self.model.head.act = torch.nn.Identity()

                # Save pointers to layers to unfreeze
                last_block_mlp  = self.model.blocks[-1].mlp
                model_norm      = self.model.norm
                model_head      = self.model.head
                if not self.args.finetune:
                    # Freeze layers
                    for param in self.model.parameters():
                        param.requires_grad = False
                    # Unfreeze saved last layers
                    for param in last_block_mlp.parameters(): param.requires_grad = True
                    for param in model_norm.parameters(): param.requires_grad = True
                    for param in model_head.parameters(): param.requires_grad = True
            else:
                raise Exception("MaskFeat must use transfer learning.") 
            # Construct last FC layer
            num_features = self.model.head.projection.in_features
            self.model.head.projection = torch.nn.Linear(num_features, self.args.num_classes)

            ## If using CORN Ordinal regression, replace final layer with -1 output nodes.
            if self.args.ordinal and self.args.ordinal_strat == 'CORN':
                num_features = self.model.head.projection.in_features
                self.model.head.projection = torch.nn.Linear(num_features, self.args.num_classes-1)
            if self.args.ordinal and self.args.ordinal_strat == 'CORAL':
                num_features = self.model.head.projection.in_features
                self.model.head.projection = CoralLayer(size_in=num_features, num_classes=self.args.num_classes)
            self.batch_key = "video"
        else:
            raise Exception(f"{self.args.arch} not supported")

    def on_train_epoch_start(self):
        # For distributed training, manually set sampler epoch to correctly shuffle
        epoch = self.trainer.current_epoch
        if self.trainer._accelerator_connector.is_distributed:
           self.trainer.datamodule.train_sampler.set_epoch(epoch)
        # Initialize training metrics
        self.train_losses = []
        self.train_accs = []
        self.train_maes = []
        # Display val_sub on progress bar
        try:
            val_subnum = int(self.args.val_sub.lower().split('sub')[-1])
            self.log("val_sub", val_subnum, prog_bar=True)
        except ValueError: None
    
    def on_train_epoch_end(self) -> None:
        # Log Histogram of model weights
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
        return super().on_train_epoch_end()

    # def on_train_end(self) -> None:
    #     # Print whether static_graph can be used
    #     ddp_logging_data = self.trainer.model._get_ddp_logging_data()
    #     print("Static graph:",ddp_logging_data.get("can_set_static_graph"))
    #     return super().on_train_end()

    def forward(self, x):
        # MaskFeat forward operates on x[0], rather than x. 
        if self.args.arch == 'mvit_maskfeat':
            x = torch.unsqueeze(x,dim=0)
        return self.model(x)
    
    """
    DEPRECATED ORDINAL CLASSIFICATION IMPLEMENTATION
    def ordinal_prediction(self, preds):
        # Convert ordinal encoding back to class labels
        ooms = [torch.floor(torch.log10(x)) for x in preds[:,0]]
        cross_thresh = torch.zeros_like(preds)
        for i, oom in enumerate(ooms):
            cross_thresh[i,:] = torch.floor(torch.log10(preds[i,:])) >= oom
        return (cross_thresh.cumprod(axis=1).sum(axis=1) - 1).int()

    def ordinal_loss(self, preds, targets):
        preds = F.softmax(preds, dim=-1)
        # Modify target with ordinal encoding
        ordinal_target = torch.zeros_like(preds)
        for i, target in enumerate(targets):
            ordinal_target[i, 0:target+1] = 1
        return F.mse_loss(preds, ordinal_target, reduction='sum')
    """
        
    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        if self.args.arch == 'mvit_maskfeat': # Maskfeat operates on x[0]
            x = torch.unsqueeze(x, dim=0)
        if self.args.consolidate:
            self.consolidate_labels(batch["label"])
        y_hat = self.model(x)
        # == LOSS == 
        if self.args.ordinal:   
            if self.args.ordinal_strat == 'CORN':
                loss = corn_loss(y_hat, batch['label'], num_classes=self.args.num_classes)
            elif self.args.ordinal_strat == 'CORAL':
                levels = levels_from_labelbatch(batch['label'], num_classes=self.args.num_classes)
                levels = levels.to(y_hat.device)
                loss = coral_loss(y_hat, levels)
            else:
                loss = self.ordinal_loss(y_hat, batch["label"])
        else:                   
            loss = F.cross_entropy(y_hat, batch["label"], label_smoothing=self.args.label_smoothing)
        # Manual logging of training metrics
        if ((batch_idx % self.args.log_every_n_steps) == 0):
            preds = F.softmax(y_hat, dim=-1)
            if self.args.ordinal:
                if self.args.ordinal_strat == 'CORN':
                    pred_labels = corn_label_from_logits(y_hat)
                elif self.args.ordinal_strat == 'CORAL':
                    pred_labels = proba_to_label(torch.sigmoid(y_hat))
                else:
                    pred_labels = self.ordinal_prediction(preds)
            else:
                pred_labels = torch.argmax(preds, dim=1)
            acc = self.train_accuracy(pred_labels, batch["label"])
            mae = self.train_MAE(pred_labels, batch["label"])
            self.train_losses.append(loss)
            self.train_accs.append(acc)
            self.train_maes.append(mae)
            avg_loss = torch.mean(torch.stack(self.train_losses))
            avg_acc = torch.mean(torch.stack(self.train_accs))
            avg_mae = torch.mean(torch.stack(self.train_maes))
            self.logger.experiment.add_scalar("Loss/Train", avg_loss, batch_idx)
            self.logger.experiment.add_scalar("Accuracy/Train", avg_acc, batch_idx)
            self.logger.experiment.add_scalar("MAE/Train", avg_mae, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        if self.args.arch == 'mvit_maskfeat': # Maskfeat operates on x[0]
            x = torch.unsqueeze(x, dim=0)
        if self.args.consolidate:
            self.consolidate_labels(batch["label"])
        y_hat = self.model(x)
        preds = F.softmax(y_hat, dim=-1)
        # Calculate loss and predicted labels
        if self.args.ordinal:   
            if self.args.ordinal_strat == 'CORN':
                loss = corn_loss(y_hat, batch['label'], num_classes=self.args.num_classes)
                pred_labels = corn_label_from_logits(y_hat)
            elif self.args.ordinal_strat == 'CORAL':
                levels = levels_from_labelbatch(batch['label'], num_classes=self.args.num_classes)
                levels = levels.to(y_hat.device)
                loss = coral_loss(y_hat, levels)
                pred_labels = proba_to_label(torch.sigmoid(y_hat))
            else:
                loss = self.ordinal_loss(y_hat, batch["label"])
                pred_labels = self.ordinal_prediction(preds)
        else:                  
            loss = F.cross_entropy(y_hat, batch["label"], label_smoothing=self.args.label_smoothing)
            pred_labels = torch.argmax(preds, dim=1)
        target = batch["label"]
        # Calculate metrics
        self.val_accuracy.update(pred_labels, target)
        self.val_MAE.update(pred_labels, target)
        # Log metrics
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log(
            "val_acc", self.val_accuracy, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log("val_MAE", self.val_MAE, on_epoch=True, sync_dist=True)
        # Custom preds, target, tasks Logging
        self.val_filenames = self.record_filenames(self.val_filenames, pred_labels, target.tolist(), batch)
        self.val_tasks = self.record_tasks(self.val_tasks, pred_labels, target.tolist(), batch['video_name'])
        self.val_preds.append(pred_labels.tolist())
        self.val_loss.append(loss.tolist())
        self.val_target.append(target.tolist())
        return loss
    
    def softmax_index(self, preds):
        for i in range (len(preds)):
            preds[i] = preds[i].index(max(preds[i]))
        return preds
    
    def record_tasks(self, tasks, preds, target, video_names):
        for i in range(len(target)): # Iterate through batch
            video_name = re.split('\d',video_names[i])[-2][1:-1]
            if video_name in tasks:
                tasks[video_name]['preds'].append(preds[i].tolist())
                tasks[video_name]['target'].append(target[i])
            else:
                tasks[video_name] = {}
                tasks[video_name]['preds'] = [preds[i].tolist()]
                tasks[video_name]['target'] = [target[i]]
        return tasks

    def record_filenames(self, filenames, preds, target, batch):
        video_names = batch['video_name']
        clip_index = batch['clip_index']
        for i in range(len(target)): # Iterate through batch
            video_name = video_names[i]
            if video_name in filenames:
                filenames[video_name]['preds'].append(preds[i].tolist())
                filenames[video_name]['target'].append(target[i])
                filenames[video_name]['clip_index'].append(clip_index[i].item())
            else:
                filenames[video_name] = {}
                filenames[video_name]['preds'] = [preds[i].tolist()]
                filenames[video_name]['target'] = [target[i]]
                filenames[video_name]['clip_index'] = [clip_index[i].item()]
        return filenames
    
    def consolidate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Consolidate 5-class labels to 3 classes. Changes labels in place. 
        Consolidated label mapping (note that GRASSP scores are label+1):
        [0, 1]  ->  [0]
        [2]     ->  [1]
        [3, 4]  ->  [2]
        """
        consolidated_mapping = [0,0,1,2,2] 
        for i, label in enumerate(labels):
            labels[i] = consolidated_mapping[label]
        
    
    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # Linear warm-up
        try:
            if self.trainer.current_epoch < self.args.warmup_epochs:
                lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / float(self.args.warmup_epochs))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.args.lr
        except AttributeError:
            if self.trainer.global_step < self.args.warmup:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.args.warmup))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.args.lr

    def configure_optimizers(self):
        if self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optim == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else: # if optim not defined, use architecture defaults. 
            if 'mvit' in self.args.arch:
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay,
                )
            else:
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                )
        min_lr = self.args.lr*1e-2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.max_epochs, eta_min=min_lr, last_epoch=-1
        )
        return [optimizer], [scheduler]
    
def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
