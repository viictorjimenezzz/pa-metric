from typing import Optional
import torch

from pametric.metrics import PosteriorAgreement
from pametric.datautils import MultienvDataset

class PosteriorAgreementDelta(PosteriorAgreement):
    """
    This class only makes sense when dataset contains a single environment. Nevertheless, we will
    allow for multiple environments, and we will remove the first one to match the second one, in a way that
    the first one contains the perfect predictions of the second one.

    PosteriorAgreement: [env0, env1, ...] then PA(0, 1) given and validated in PA(0, ...)
    PosteriorAgreementDelta: [env0, env1, ...] then PA(1_perfect, 1) given and validated in PA(1_perfect, ...)
    """
    def __init__(pairing_strategy: Optional[str] = None, *args, **kwargs):
        """
        Pairing strategy must not be selected by any means, so it will not be passed further (default to None)
        """

        dataset = kwargs["dataset"]
        if dataset.num_envs == 1:
            kwargs["dataset"] = MultienvDataset([dataset.dset_list[0]]*2)
        else:
            kwargs["dataset"] = MultienvDataset([dataset.dset_list[1]] + dataset.dset_list[1:])

        super().__init__(*args, **kwargs)

    def _generate_logits(self, y: torch.Tensor, n_classes: int) -> torch.Tensor:
        sample_size = y.size(0)
        logits = torch.zeros((sample_size, n_classes), dtype=torch.float)
        logits[torch.arange(sample_size), y] = 1.0
        return logits
    
    def _get_logits_from_batch(self, rank: int, batch, env_logit1: Optional[str] = "1"):
        dev = self._get_current_dev(rank)

        envs = list(batch.keys())
        y = batch[envs[0]][1]
        logits1 = batch[str(env_logit1)][0]
        logits0 = self._generate_logits(y, logits1.size(1))
        
        return logits0.to(dev), logits1.to(dev), y.to(dev)
    
