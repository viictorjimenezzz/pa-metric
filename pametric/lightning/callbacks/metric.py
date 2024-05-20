from typing import Optional, List, Union
from copy import deepcopy

import torch
import torch.nn.functional as F

# If the metric is to be implemented in a lightning training procedure.
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from pametric.metrics import PosteriorAgreement, PosteriorAgreementDelta
from pametric.datautils import MultienvDataset, LogitsDataset

class PA_Callback(Callback):
    def __init__(
            self,
            log_every_n_epochs: int,
            dataset: MultienvDataset,
            pa_epochs: int,
            deltametric: Optional[bool] = False,
            preds_2_factor: Optional[float] = 1.0,
            beta0: Optional[float] = 1.0,
            pairing_strategy: Optional[str] = None,
            pairing_csv: Optional[str] = None,
            feature_extractor: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            cuda_devices: Optional[Union[List[str], int]] = 0,
            batch_size: Optional[int] = 16,
            num_workers: Optional[int] = 0,
        ):
        super().__init__()

        # Check dataset conditions: we use .__name__ because the class is defined twice (for librarization)
        # assert dataset.__class__.__name__ in ["MultienvDataset"], "The dataset must be an instance of MultienvDataset."

        self.beta0 = beta0
        self.pa_epochs = pa_epochs
        self.log_every_n_epochs = log_every_n_epochs

        # Be able to substitute the pl_moduke.model by another model on the fly
        self.alternative_model = None

        self.deltametric = deltametric
        self.pa_metric = self._posterioragreementmetric(
                            dataset = dataset,
                            pa_epochs = self.pa_epochs,
                            beta0 = self.beta0,
                            preds_2_factor = preds_2_factor,
                            pairing_strategy = pairing_strategy,
                            pairing_csv = pairing_csv,
                            feature_extractor = feature_extractor,
                            processing_strategy = "lightning",
                            optimizer = optimizer,
                            cuda_devices = cuda_devices,
                            batch_size = batch_size,
                            num_workers = num_workers
        )

    def _posterioragreementmetric(self, *args, **kwargs):
        if self.deltametric:
            return PosteriorAgreementDelta(*args, **kwargs)
        return PosteriorAgreement(*args, **kwargs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if (pl_module.current_epoch + 1) % self.log_every_n_epochs == 0:     
            pa_dict = self.pa_metric(
                classifier=pl_module.model if self.alternative_model is None else self.alternative_model,
                local_rank=trainer.local_rank,
                # TODO: Consider there's early_stopping in the pl_module. How can I fix that?
                destroy_process_group = False
            )
            for env_index, metric_dict in pa_dict.items():
                dict_to_log = {
                    f"PA(0,{env_index+1})/beta": metric_dict["beta"],
                    f"PA(0,{env_index+1})/logPA": metric_dict["logPA"],
                    f"PA(0,{env_index+1})/AFR_pred": metric_dict["AFR_pred"],
                    f"PA(0,{env_index+1})/AFR_true": metric_dict["AFR_true"],
                    f"PA(0,{env_index+1})/acc_pa": metric_dict["acc_pa"]
                }
                self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # Build gradient graph for the model
        test_model = deepcopy(pl_module.model)
        for param in test_model.parameters():
            param.requires_grad = True

        pa_dict = self.pa_metric(
            classifier=test_model,
            local_rank=trainer.local_rank,
            destroy_process_group = False
        )
        for env_index, metric_dict in pa_dict.items():
            dict_to_log = {
                f"PA(0,{env_index+1})/beta_test": metric_dict["beta"],
                f"PA(0,{env_index+1})/logPA_test": metric_dict["logPA"],
                f"PA(0,{env_index+1})/AFR_pred_test": metric_dict["AFR_pred"],
                f"PA(0,{env_index+1})/AFR_true_test": metric_dict["AFR_true"],
                f"PA(0,{env_index+1})/acc_pa_test": metric_dict["acc_pa"]
            }
            self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)