from pytorch_lightning.callbacks import BatchSizeFinder
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.utilities.exceptions import _TunerExitException

from typing import Optional
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader

class MultienvBatchSizeFinder(BatchSizeFinder):
    """
    Extension of pytorch_lightning BatchSizeFinder to allow for CombinedLoader or dictionaries
    of Dataloaders as outputs of the LightningDataModule. Two main variations from the original are implemented:
        - The final batch_size will be batch_size_OOM // num_envs.
        - If a list of dataloaders (or a CombinedLoader) is given, the limit batch size will be 
            found for the first one.

    This variation will allow us to send the whole batch to the GPU.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        """
            Compute number of environments depending on the type of dl_source.dataloader()
        """
        super().setup(trainer, pl_module, stage)

        dl_source = getattr(trainer._data_connector, f"_{trainer.state.stage.dataloader_prefix}_dataloader_source")
        dl = dl_source.dataloader()
        if isinstance(dl, dict):
            self._num_envs = dl.keys().__len__()
        elif isinstance(dl, CombinedLoader):
            self._num_envs = next(iter(dl)).keys().__len__()
        else:
            self._num_envs = 1

    def scale_batch_size(self, trainer: Trainer, pl_module: LightningModule) -> None:
        new_size = scale_batch_size(
            trainer,
            pl_module,
            self._mode,
            self._steps_per_trial,
            self._init_val,
            self._max_trials,
            self._batch_arg_name,
        )

        self.optimal_batch_size = new_size // self._num_envs
        if self._early_exit:
            raise _TunerExitException()

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        We keep the same batch size during validation.
        """
        pass