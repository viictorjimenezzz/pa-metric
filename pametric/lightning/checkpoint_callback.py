
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


class PA_ModelCheckpoint(ModelCheckpoint):
    """
    Simple subclass of ModelCheckpoint to save the best model based on the PA metric. Since
    the pairing of observations is often performed based on the features extracted by the model
    at epoch zero, it is likely that the highest PA is achieved at the first epoch. To avoid that, 
    it is recommended to select the model that achieves the maximum PA after the first minimum 
    has been achieved.

    Since the selection might depend on the problem (often avoiding `epoch=0` is enough), we will 
    include the parameter `patience` to allow the user to select the epoch to start looking for the
    best model.

    Other parameters inherited from PytorchLightning ModelCheckpoint will be set to the required
    values for PA model selection to facilitate the user's experience. 
    """

    def __init__(self, patience: int = 1, **kwargs):

        """
        Args:
            patience (int): Number of epochs to wait after the first minimum has been achieved.
        """

        # Overriding default/user values to the specific requirements of PA model selection:
        kwargs['mode'] = "max"
        kwargs['every_n_train_steps'] = None
        kwargs["save_on_train_epoch_end"] = None

        super().__init__(**kwargs)
        self.patience = patience

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        if trainer.current_epoch < self.patience:
            return True
        else:
            return super()._should_skip_saving_checkpoint(trainer)
        