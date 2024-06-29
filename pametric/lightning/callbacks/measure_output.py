from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

import numpy as np
from typing import Optional
from copy import deepcopy
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

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

    def __init__(self, average: bool = True, pametric_callback_name: Optional[str] = "PA_Callback"):
        super().__init__()
        self.average = average
        self.pametric_callback_name = pametric_callback_name
    
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
        dataloader = self._reinstantiate_dataloader(dataloader)
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

    def _reinstantiate_dataloader(self, dataloader: DataLoader):
        return DataLoader(
            dataset=dataloader.dataset,
            collate_fn=MultiEnv_collate_fn,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            sampler=SequentialSampler(dataloader.dataset),
            drop_last=False,
        )

    def _compute_average(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        # Get the model and split it into feature extractor and classifier
        model_to_eval = SplitClassifier(
            net = deepcopy(pl_module.model.net),
            net_name = pl_module.model.net_name
        ).eval()

        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pa_metric_callback = trainer.callbacks[callback_names.index(self.pametric_callback_name)]
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
    
    def _log_average(self, average_val: torch.Tensor, metric_name:  Optional[str] = None, log: Optional[bool] = True) -> None:
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


class CentroidDistance_Callback(MeasureOutput_Callback):

    """
    Computes l_p distance between centroids of feature spaces of each domain.

    Args:
        p_dist (float): The norm to be used. Defaults to l_inf.
        by_label (bool): Whether a single centroid is computed for each environment or 
        a centroid is computed for each label, and compared between environments. This would give us a centroid
        distance between samples of the same class, which is actually informative for PA.
    """

    metric_name: str = "CD"
    output_features = True

    def __init__(self, p_dist: float = float("inf"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_dist = p_dist
        self.average = False

    def _compute_average(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        self.num_classes = trainer.datamodule.num_classes
        return super()._compute_average(trainer, pl_module)
    
    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: torch.nn.Module) -> torch.Tensor:
        dataloader = self._reinstantiate_dataloader(dataloader)

        sum_features = [None]*self.num_classes
        for bidx, batch in enumerate(dataloader):
            list_envs = list(batch.keys())
            # Here depends wether the features have to be extracted or not
            output = [
                model_to_eval.forward(batch[e][0], self.output_features)
                for e in list_envs
            ]
            # Assume label-correspondence (i.e. same labels in each environment)
            labels = batch[list_envs[0]][1]
            unique_labels, counts = torch.unique(labels, return_counts=True)
            if bidx == 0:
                label_counts_dict = dict(zip(unique_labels.tolist(), counts.tolist()))
            else:
                for ilab, lab in enumerate(unique_labels):
                    label_counts_dict[lab.item()] += counts[ilab].item()

            for lab in range(self.num_classes):
                mask = labels == lab

                if bidx == 0:
                    sum_features[lab] = [
                        out_e[mask, :].sum(dim=0) for out_e in output
                    ]
                else:
                    for e, out_e in enumerate(output):
                        sum_features[lab][e] += out_e.sum(dim=0)

        centers = [
            [
                sum_features[lab][e] / label_counts_dict[lab]
                for e in range(self.num_envs)
            ]
            for lab in range(self.num_classes)
        ]
        centers_dist_lab = [
            [
                self._cdist_centers(centers[lab][0], centers[lab][e+1])
                for e in range(self.num_envs-1)
            ]
            for lab in range(self.num_classes)
        ]

        # Then add the sums for each label:
        sum_features = [
            torch.stack([
                sum_features[lab][e] for lab in range(self.num_classes)
            ]).sum(dim=0)
            for e in range(self.num_envs)
        ]
        centers = [sum_features[e] / self.len_dataset for e in range(self.num_envs)]
        centers_dist = torch.tensor([
            self._cdist_centers(centers[0], centers[e+1])
            for e in range(self.num_envs-1)
        ])
        return (centers_dist, centers_dist_lab)
    
    def _cdist_centers(self, center0: torch.Tensor, center1: torch.Tensor) -> float:
        return torch.cdist(center0.unsqueeze(0).unsqueeze(0), center1.unsqueeze(0).unsqueeze(0), p=self.p_dist).item()
    
    def _log_average(self, distance_tuple: tuple, metric_name:  Optional[str] = None, log: Optional[bool] = True) -> None:
        centers_dist, centers_dist_lab = distance_tuple

        dict_to_log = {
            f"PA(0,{e+1})/CD_{lab_ind}": dist_lab[e]
            for lab_ind, dist_lab in enumerate(centers_dist_lab)
            for e in range(self.num_envs-1)
        }

        dict_to_log.update({
            f"PA(0,{e+1})/CD": centers_dist[e].item()
            for e in range(self.num_envs-1)
        })

        if log:
            self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return dict_to_log


class FrechetInceptionDistance_Callback(MeasureOutput_Callback):
    """
    The frechet inception distance (FID) between images is used to assess the quality in generative models. It amounts to
    the 2D-Wasserstein distance between normal distributions fitted to the feature representation of images.
    """

    metric_name: str = "FID"
    output_features = True # The argument of _metric are the feature vectors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.average = False

    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: torch.nn.Module) -> torch.Tensor:
        dataloader = self._reinstantiate_dataloader(dataloader)

        output = []
        for bidx, batch in enumerate(dataloader):
            list_envs = list(batch.keys())
            for ind_e, e in enumerate(list_envs):
                if bidx == 0:
                    output.append([model_to_eval.forward(batch[e][0], self.output_features)])
                else:
                    output[ind_e].append(
                        model_to_eval.forward(batch[e][0], self.output_features)
                    )

        output = [torch.cat(out_e, dim=0) for out_e in output]
        
        fid_values = []
        mu_0, sigma_0 = self._statistics(output[0])
        for e in range(self.num_envs-1):
            mu_e, sigma_e = self._statistics(output[e+1])
            fid_values.append(
                self._compute_fid(mu_0, sigma_0, mu_e, sigma_e)
            )
        return torch.tensor(fid_values)
    
    def _statistics(self, feature_matrix: torch.Tensor):
        mu = torch.mean(feature_matrix, dim=0)
        sigma = torch.from_numpy(np.cov(feature_matrix.T.cpu().numpy()))
        return mu, sigma

    def _compute_fid(self, mu_0, sigma_0, mu_1, sigma_1):
        diff = mu_0 - mu_1
        covmean = sqrtm(sigma_0.cpu().numpy() @ sigma_1.cpu().numpy())

        # Check if covmean is complex, and if so, convert to real
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + torch.trace(sigma_0 + sigma_1 - 2*torch.from_numpy(covmean))
        return fid.item()
    

class MMD_Callback(MeasureOutput_Callback):

    metric_name: str = "MMD"
    output_features = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.average = False

    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: torch.nn.Module) -> torch.Tensor:
        dataloader = self._reinstantiate_dataloader(dataloader)

        output = []
        for bidx, batch in enumerate(dataloader):
            list_envs = list(batch.keys())
            for ind_e, e in enumerate(list_envs):
                if bidx == 0:
                    output.append([model_to_eval.forward(batch[e][0], self.output_features)])
                else:
                    output[ind_e].append(
                        model_to_eval.forward(batch[e][0], self.output_features)
                    )

        output = [torch.cat(out_e, dim=0) for out_e in output]
        
        mmd_values = []
        for e in range(self.num_envs-1):
            mmd_values.append(
                self._compute_mmd(output[0], output[e+1])
            )

        return mmd_values
    
    def _compute_mmd(self, x, y):
        # https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
        """
            Emprical maximum mean discrepancy. The lower the result
            the more evidence that distributions are the same.
        """
        dev = x.device
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(dev),
                    torch.zeros(xx.shape).to(dev),
                    torch.zeros(xx.shape).to(dev))
        
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

        return torch.mean(XX + YY - 2. * XY)