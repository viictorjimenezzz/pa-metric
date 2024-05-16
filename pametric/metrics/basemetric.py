from typing import Optional, Any
from tqdm import tqdm

import torch
from torchmetrics import Metric
from torch.utils.data import DataLoader

from copy import deepcopy

from pametric.datautils import MultienvDataset, MultiEnv_collate_fn
from pametric.pairing import PosteriorAgreementDatasetPairing
from pametric.kernel import PosteriorAgreementKernel

class PosteriorAgreementBase(Metric):

    r"""Performs the optimization of the empirical Posterior Agreement kernel for a discrete hypothesis class.

    .. math::
        \begin{aligned}
        \operatorname{PA}\left(X^{\prime}, X^{\prime \prime}\right)=\underset{\beta}{\operatorname{maximize}} & \frac{1}{N} k\left(X^{\prime}, X^{\prime \prime}\right) . \\
        \text { subject to } & \beta \geq 0
        \end{aligned}

    The expression of the kernel is the following:

    .. math::
        k\left(X^{\prime}, X^{\prime \prime}\right)=\log \left(\sum_{c \in \mathcal{C}} \frac{p\left(c \mid X^{\prime}\right) p\left(c \mid X^{\prime \prime}\right)}{p(c)}\right)

    Where :math:`p\left(c \mid X\right)` is the posterior probability distribution over the hypothesis space :math:`\mathcal{C}` given the data :math:`X`. Its expression corresponds to a Gibbs distribution:

    .. math::
        p(c \mid X)=\frac{\exp (\beta R(c, X))}{\exp (1+\alpha)}

    where :math:`\beta` is the inverse temperature parameter.

    Using the metric is straightforward. It only requires initialization with an appropiate MultienvDataset and then can be
    called with the classifiers to evaluate. The metric will perform the optimization and return the maximum PA value.

    Example:

        >>> from posterioragreement.metrics import PosteriorAgreementBase
        >>> from posterioragreement.datautils import MultienvDataset
        >>> ds = MultienvDataset([ds_env1, ds_env2])
        >>> pa = PosteriorAgreementBase(ds, pa_epochs=10)
        >>> pa(classifier)["logPA"]
        tensor(-10.5000)

    The structure of the computation for each of these methods is the following:

    __init__():
        . Perform pairing of the dataset (if required).
        . _multiprocessing_conf()
	

    forward():
        . update():
            . _initialize_classifiers()
            . Preallocates metrics on the cpu.	
            . pa_update(): 
                . _initialize_optimization()
                . Initialize image dataloader.
                for epoch in range(pa_epochs):
                    . pa_optimize()
                    . pa_evaluate()
                    . Store required metrics
            . _delete_classifiers()
        
        . compute():
            . Select index of maximum PA.
            . Retrieve the associated metrics.
    """

    def __init__(
            self,
            dataset: MultienvDataset,
            pa_epochs: int,
            beta0: Optional[float] = 1.0,
            preds_2_factor: Optional[float] = 1.0,
            pairing_strategy: Optional[str] = "label",
            pairing_csv: Optional[str] = None,
            feature_extractor: Optional[torch.nn.Module] = None
        ):
        """
        Initialization args:
            - `dataset` (MultienvDataset): Multienvironment dataset instance. Can be initialized with a list of datasets as `MultienvDataset([ds1, ds2, ...])`.
            - `pa_epochs` (int): Number of epochs to optimize the kernel at every .update() call.
            - `beta0` (float): Initial value of the beta parameter.
            - `pairing_strategy` (str): Strategy to pair the observations. Can be label-corresponding "label" or nearest-neighbour "nn". Defaults to "label".
            - `pairing_csv` (str): Path to a csv file containing the pairing of the dataset. If no file exists, the pairing will be performed and then stored at the specified location.
        """
        super().__init__()

        self.beta0 = beta0 # initial value of the beta parameter
        self.pa_epochs = pa_epochs # number of epochs to optimize the kernel at every .update() call
        self.preds_2_factor = preds_2_factor

        # (Multi?)processing strategy
        self._multiprocessing_conf() # defines device_list and ddp_init

        assert isinstance(dataset, MultienvDataset), "The dataset must be an instance of MultienvDataset or LogitsDataset."
        self.pairing_strategy = pairing_strategy
        self.dataset = PosteriorAgreementDatasetPairing(
            dataset,
            pairing_strategy,
            pairing_csv,
            feature_extractor
        ) # make sure dataset is paired
        self.num_envs = self.dataset.num_envs

    def _multiprocessing_conf(self): 
        """
        Override with the desired logic for defining the variables `device_list` and `ddp_init`.
        """
        self.device_list = ["cpu" if torch.cuda.is_available() else "cpu"]
        self.ddp_init = None

    def _get_world_size(self):
        """
        Override with the logic for retrieving the world size.
        """
        return 1

    def _get_current_dev(self, rank: int):
        """
        Override with the desired device definition based on current (local) rank.
        """
        return self.device_list[rank]

    def _initialize_optimization(self, rank: int):
        """
        Override with the desired logic for defining the PA kernel and the optimizer.
        """
        dev = self._get_current_dev(rank)
        kernel = PosteriorAgreementKernel(beta0=self.beta0, preds_2_factor=self.preds_2_factor).to(dev)
        optimizer = torch.optim.Adam([kernel.module.beta], lr=0.1)
        return kernel, optimizer
    
    def _initialize_classifiers(self, classifier: torch.nn.Module, classifier_val: Optional[torch.nn.Module] = None):
        """
        Deepcopy and freeze input classifiers.
        """
        classifier = deepcopy(classifier).eval()
        for param in classifier.parameters():
            param.requires_grad = False
        if classifier_val:
            classifier_val = deepcopy(classifier_val).eval()
            for param in classifier_val.parameters():
                param.requires_grad = False

        return classifier, classifier_val

    def _delete_classifiers(self):
        """
        Delete classifiers from memory when they are no longer needed. Override with desired teardown strategy.
        """
        del self.classifier
        del self.classifier_val

    def _get_logits_from_batch(self, rank: int, batch: dict):
        """
        Override with the desired logic for extracting the logits and the preds from the batch.
        """
        dev = self._get_current_dev(rank)

        envs = list(batch.keys())
        logits0, logits1 = self.classifier(batch[envs[0]][0].to(dev)), self.classifier_val(batch[envs[1]][0].to(dev))
        y = batch[envs[0]][1].to(dev)
        return logits0, logits1, y

    def pa_optimize(self, rank: int):
        """
        Performs a beta optimization epoch.
        """
        for bidx, batch in enumerate(self.pa_dataloader):
            self.kernel.module.beta.data.clamp_(min=0.0)
            self.kernel.module.reset()
            self.kernel.module.train()

            logits0, logits1, _ = self._get_logits_from_batch(rank, batch)
            with torch.set_grad_enabled(True):
                loss = self.kernel.module.forward(logits0, logits1)  
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.kernel.module.beta.data.clamp_(min=0.0) # project to >=0 one last time
        return self.kernel.module.beta.item()

    def pa_evaluate(self, rank: int, beta: float, is_first_epoch: Optional[bool] = False):
        """
        Performs an evaluation epoch for a fixed beta to find the PA value.
        """
        self.kernel.module.reset()
        self.kernel.module.eval()

        correct, correct_pred, correct_true = 0, 0, 0
        for bidx, batch in enumerate(self.pa_dataloader):
            logits0, logits1, y = self._get_logits_from_batch(rank, batch)

            # Compute accuracy metrics only once
            if is_first_epoch:
                y_pred = torch.argmax(logits0, 1) # env 1
                y_pred_adv = torch.argmax(logits1, 1) # env2 or y
                correct_pred += (y_pred_adv == y_pred).sum().item()
                correct_true += (y_pred_adv == y).sum().item()
                correct += (torch.cat([y_pred, y_pred_adv]) == torch.cat([y, y])).sum().item()

            # Update logPA
            self.kernel.module.evaluate(logits0, logits1, beta)

        return {
                "logPA": self.kernel.module.log_posterior().item(),
                "AFR_pred": correct_pred/(len(logits0)*(bidx+1)),
                "AFR_true": correct_true/(len(logits0)*(bidx+1)),
                "accuracy": correct/(2*len(logits0)*(bidx+1))
            }  if is_first_epoch else {"logPA": self.kernel.module.log_posterior().item()}


    def pa_update(self, rank: int):
        """
        Calls `pa_optimize()` and `pa_evaluate()` for `pa_epochs` iterations, and stores the appropiate results.
        It also manages model and tensor allocation in devices.
        """
        # Initialize the kernel and optimizer
        self.kernel, self.optimizer = self._initialize_optimization(rank)

        # Initialize the dataloader
        self.pa_dataloader = DataLoader(
            dataset=self.dataset,
            collate_fn=MultiEnv_collate_fn,

            batch_size = 16,
            num_workers = 0, 

            pin_memory = ('cuda' in self.device_list[0]),
            drop_last=False,
            shuffle=False,
        )

        # In the basemetric, only one device is used.
        dev = self._get_current_dev(rank)
        self.classifier.to(dev)
        self.classifier_val.to(dev)

        # For every epoch, optimize beta and then evaluate PA with the last beta
        for epoch in tqdm(range(self.pa_epochs)):
            self.betas[epoch] = self.pa_optimize(rank)

            metrics_dict = self.pa_evaluate(rank, self.betas[epoch].item(), is_first_epoch = (epoch == 0))
            self.logPAs[epoch] = metrics_dict["logPA"]

            # Store accuracy metrics
            if epoch == 0:
                self.afr_pred = metrics_dict["AFR_pred"]
                self.afr_true = metrics_dict["AFR_true"]
                self.accuracy = metrics_dict["accuracy"]

    def update(self, classifier: torch.nn.Module, classifier_val: Optional[torch.nn.Module] = None):
        """
        Updates the metric with a new posterior agreement evaluation for the given classifiers. At least one classifier is required to evaluate the data. 
        If a `classifier_val` is provided, then `classifier` will evaluate the first environment, and `classifier_val` the rest.

        Args:
            - `classifier` (torch.nn.Module): Classifier to evaluate the first environment.
            - `classifier_val` (torch.nn.Module): Classifier to evaluate the rest of the environments. If not provided, `classifier` will be used.
        """
        # Initialize classifiers
        self.classifier, self.classifier_val = self._initialize_classifiers(classifier, classifier_val)
        if not self.classifier_val:
            self.classifier_val = deepcopy(self.classifier)

        # Preallocate vector of betas and logPAs in the main device
        self.betas = torch.zeros(self.pa_epochs, dtype=torch.float64).to("cpu")
        self.logPAs = torch.full_like(self.betas, -float('inf')).to("cpu")

        # Preallocate metrics in the main device
        self.afr_true = torch.tensor(0.0).to("cpu")
        self.afr_pred = torch.tensor(0.0).to("cpu")
        self.accuracy = torch.tensor(0.0).to("cpu")

        # Update beta and compute new logPA.
        self.pa_update(rank = 0) # Only one device in basemetric
        self._delete_classifiers() # free some memory

    def compute(self) -> dict:
        # Select the epoch achieving maximum logPA
        self.selected_index = torch.argmax(self.logPAs).item()

        return {
            "beta": self.betas[self.selected_index].item(),
            "logPA": self.logPAs[self.selected_index].item(),
            "PA": torch.exp(self.logPAs[self.selected_index]).item(),
            "AFR_pred": self.afr_pred,
            "AFR_true": self.afr_true,
            "acc_pa": self.accuracy
        }
    
    def forward(self, *args: Any, **kwargs: Any) -> dict:
        """
        Call .update() and .compute() in a single call.
        """
        self.update(*args, **kwargs)
        return self.compute()