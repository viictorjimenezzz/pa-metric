import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

class PosteriorAgreementKernel(nn.Module):
    def __init__(self, beta0: Optional[float] = None, device: str = "cpu"):
        super().__init__()
        beta0 = beta0 if beta0 else 1.0
        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        
        self.dev = device
        self.beta = torch.nn.Parameter(torch.tensor([beta0], dtype=torch.float), requires_grad=True).to(self.dev)
        self.log_post = torch.tensor([0.0], requires_grad=True).to(self.dev)

    def _compute_pa(self, preds1, preds2, beta_1: Optional[float] = None):
        beta = self.beta if beta_1 is None else beta_1

        probs1 = F.softmax(beta * preds1, dim=1).to(self.dev)
        probs2 = F.softmax(beta * preds2, dim=1).to(self.dev)

        return (probs1 * probs2).sum(dim=1).to(self.dev)

    def forward(self, preds1, preds2):
        self.beta.requires_grad_(True)
        self.beta.data.clamp_(min=0.0)
        self.reset()

        with torch.set_grad_enabled(True):
            probs_sum = self._compute_pa(preds1, preds2)

            # log correction for numerical stability: replace values less than eps
            # with eps, in a gradient compliant way. Replace nans in gradients
            # deriving from 0 * inf
            probs_sum = probs_sum + (probs_sum < 1e-44) * (1e-44 - probs_sum)
            if probs_sum.requires_grad:
                probs_sum.register_hook(torch.nan_to_num)
        
            self.log_post = self.log_post + torch.log(probs_sum).sum(dim=0).to(self.dev)
            return -self.log_post

    def evaluate(self, preds1, preds2, beta_fixed):
        with torch.set_grad_enabled(False):
            probs_sum = self._compute_pa(preds1, preds2, beta_fixed)
            self.log_post = self.log_post + torch.log(probs_sum).sum(dim=0).to(self.dev)
    
    def reset(self):
        self.log_post = torch.tensor([0.0], requires_grad=True).to(self.dev)

    def log_posterior(self):
        return self.log_post.clone().to(self.dev)

    def posterior(self):
        return torch.exp(self.log_post).to(self.dev)
    
    @property
    def module(self):
        """Returns the kernel itself. It helps the kernel be accessed in both DDP and non-DDP mode."""
        return self


class PosteriorAccuracyKernel(PosteriorAgreementKernel):
    def __init__(
        self,
        sharpness_factor: float,
        *args, **kwargs
    ):  
        assert sharpness_factor > 1.0, "Sharpness factor must be greater than 1."
        super().__init__(*args, **kwargs)

        self.sharpness_factor = torch.tensor([sharpness_factor], dtype=torch.float).to(self.dev)

    def _scaled_gibbs(self, v_ones, y):
        v_ones[torch.arange(len(y)), y] = self.sharpness_factor
        preds_gibbs = F.softmax(v_ones, dim=-1)
        return preds_gibbs

    def _compute_pa(self, preds1, y, beta_1: Optional[float] = None):
        beta = self.beta if beta_1 is None else beta_1

        probs1 = F.softmax(beta * preds1, dim=1).to(self.dev)
        probs2 = self._scaled_gibbs(torch.ones_like(preds1), y)
        return (probs1 * probs2).sum(dim=1).to(self.dev)