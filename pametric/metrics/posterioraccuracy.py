from typing import Optional, List

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from pametric.kernel import PosteriorAccuracyKernel
from pametric.metrics import PosteriorAgreement

class PosteriorAccuracy(PosteriorAgreement):

    """
    Equivalent metric that computes the Posterior Agreement between 
    """
    def __init__(
        self,
        sharpness_factor: float,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.num_envs == 1, "The PosteriorAccuracy metric only works with one environment."

        self.accuracymetric = True
        self.sharpness_factor = sharpness_factor        

    def _initialize_optimization(self, dev):
        """We initialize the PosteriorAccuracyKernel instead, with the additional sharpness_factor."""
        kernel = PosteriorAccuracyKernel(
            beta0=self.beta0,
            sharpness_factor=self.sharpness_factor
        ).to(dev)

        if "cuda" in dev and self.processing_strategy == "cuda":
            kernel = DDP(kernel, device_ids=[dev])
        optimizer = self.partial_optimizer([kernel.module.beta]) if self.partial_optimizer else torch.optim.Adam([kernel.module.beta], lr=0.1)
        return kernel, optimizer
    
    def _get_logits_from_batch(self, batch, env_logit1: Optional[str] = "1"):
        envs = list(batch.keys())
        logits0, y = batch[envs[0]][0], batch[envs[0]][1] # Here is the same as in PA

        # There is no environment 2, so I will return y again, so that the kernel can compute the gibbs distribution.
        return logits0, y, y
    
    def pa_update(
            self,
            rank: int,
            classifier: torch.nn.Module,
            classifier_val: Optional[torch.nn.Module] = None
        ):
        """Ensure no classifier_val is passed."""
        super().pa_update(rank, classifier, None)