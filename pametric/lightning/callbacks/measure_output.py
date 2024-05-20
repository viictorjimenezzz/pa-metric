from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

from copy import deepcopy
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from pametric.datautils import MultiEnv_collate_fn

from pametric.lightning import SplitClassifier

class MeasureOutput_Callback(Callback):
    """
    Base callback to compute metrics associated with the output of the model
    wrt the data in the PA datasets. 
    """
    metric_name: str = "metric_name"
    output_features: bool = False # Determine whether the metric works with features or with logits

    def __init__(self, average: bool = True):
        super().__init__()
        self.average = average
    
    def _metric(self, out_1: torch.Tensor, out_2: torch.Tensor) -> float:
        """
        To override with the computation of the metric.
        """
        pass    

    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: nn.Module) -> torch.Tensor:
        """
        Iterates over the dataloader to compute the aggregated metric. It is detached from
        `_compute_average()` so that it can be overriden to include more than one metric at the
        same time.
        """  
        sum_val = torch.zeros(self.num_envs-1)
        for _, batch in enumerate(dataloader):
            # Here depends wether the features have to be extracted or not
            output = [
                model_to_eval.forward(batch[e][0], self.output_features)
                for e in list(batch.keys())
            ]
            
            sum_val += torch.tensor([
                self._metric(output[e], output[e+1])
                for e in range(self.num_envs-1)
            ]) 
        return sum_val

    def _compute_average(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        # Get the model and split it into feature extractor and classifier
        model_to_eval = SplitClassifier(
            net = deepcopy(pl_module.model.net),
            net_name = pl_module.model.net_name
        ).eval()

        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pa_metric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
        dataset = pa_metric_callback.pa_metric.dataset
        self.num_envs = dataset.num_envs
        self.len_dataset = len(dataset)

        dataloader = DataLoader(
            dataset = dataset,
            collate_fn = MultiEnv_collate_fn,
            batch_size = pa_metric_callback.pa_metric.batch_size,
            num_workers = 0, 
            pin_memory = False,
            sampler = SequentialSampler(dataset),
            drop_last=False,
        )

        with torch.no_grad():
            sum_val = self._iterate_and_sum(dataloader, model_to_eval)
            if self.average:
                return sum_val / len(dataset)
            return sum_val
    
    def _log_average(self, average_val: torch.Tensor, metric_name: str, log: bool = True) -> None:
        metric_name = metric_name if metric_name is not None else self.metric_name
        dict_to_log = {
            f"PA(0,{e+1})/{metric_name}": average_val[e].item()
            for e in range(self.num_envs-1)
        }
        if log:
            self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return dict_to_log
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        average_val = self._compute_average(trainer=trainer, pl_module=pl_module)
        self._log_average(average_val)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        average_val = self._compute_average(trainer=trainer, pl_module=pl_module)
        self._log_average(average_val)


class KL_Callback(MeasureOutput_Callback):
    """
    KL is the Kullback-Leibler divergence between probability distributions.
    """

    metric_name: str = "KL"
    output_features = False # The argument of _metric are the logits

    def _metric(self, out_1: torch.Tensor, out_2: torch.Tensor) -> float:
        log_softmax1 = F.log_softmax(out_1, dim=1)
        softmax2 = F.softmax(out_2, dim=1)  # Use softmax, not log_softmax
        return F.kl_div(log_softmax1, softmax2, reduction='batchmean').item()

class Wasserstein_Callback(MeasureOutput_Callback):
    """
    The wasserstein distance between two probability distributions.
    """

    metric_name: str = "W"
    output_features = False # The argument of _metric are the logits

    def _metric(self, out_1: torch.Tensor, out_2: torch.Tensor) -> float:
        probs1_np = F.softmax(out_1, dim=1).detach().cpu().numpy().flatten()  # Flatten to 1D
        probs2_np = F.softmax(out_2, dim=1).detach().cpu().numpy().flatten()
        return wasserstein_distance(probs1_np, probs2_np)
    
class CosineSimilarity_Callback(MeasureOutput_Callback):
    """
    The cosine similarity between feature vectors.
    """

    metric_name: str = "CS"
    output_features = True # The argument of _metric are the feature vectors

    def _metric(self, out_1: torch.Tensor, out_2: torch.Tensor) -> float:
        return F.cosine_similarity(out_1, out_2, dim=1).sum().item()