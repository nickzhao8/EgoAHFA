from pytorch_lightning.callbacks import Callback

class DebugCallback(Callback):
    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_epoch_start(trainer, pl_module)
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_epoch_end(trainer, pl_module)
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_start(trainer, pl_module)
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_end(trainer, pl_module)
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_validation_start(trainer, pl_module)
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_validation_end(trainer, pl_module)
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_fit_start(trainer, pl_module)
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_fit_end(trainer, pl_module)