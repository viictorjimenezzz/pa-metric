import os
import warnings
import gc

import torch
from typing import Any, Optional, List, Union
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from pametric.datautils import MultienvDataset, LogitsDataset, MultiEnv_collate_fn, adjust_batch_size
from pametric.pairing import PosteriorAgreementDatasetPairing
from pametric.kernel import PosteriorAgreementKernel, PosteriorAccuracyKernel

from pametric.metrics import PosteriorAgreementBase

class PosteriorAgreement(PosteriorAgreementBase):

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

    `PosteriorAgreement` adds some functionalities to `PosteriorAgreementBase` that make the computation of the metric more efficient:
        - It allows for the optimization of the PA metricto be parallelized (DDP) to multiple GPUs, using the `processing_strategy` parameter.
        - It evaluates the classifiers only once per call, and then uses the same logits to optimize beta for several epochs.

    Using the metric is straightforward. It only requires initialization with an appropiate MultienvDataset and then can be
    called with the classifiers to evaluate. The metric will perform the optimization and return the maximum PA value.

    Example:

        >>> from posterioragreement.metrics import PosteriorAgreementBase
        >>> from posterioragreement.datautils import MultienvDataset
        >>> ds = MultienvDataset([ds_env1, ds_env2])
        >>> pa = PosteriorAgreement(ds, pa_epochs=10)
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
        optimizer: Optional[torch.optim.Optimizer] = None,
        processing_strategy: Optional[str] = "cpu",
        cuda_devices: Optional[Union[List[str], int]] = [0],
        batch_size: Optional[int] = 16, # for the images
        num_workers: Optional[int] = 0, # for the images
        *args, **kwargs
    ):
        """
        Initialization args common to `PosteriorAgreementBase`:
            - `dataset` (MultienvDataset): Multienvironment dataset instance. Can be initialized with a list of datasets as `MultienvDataset([ds1, ds2, ...])`.
            - `pa_epochs` (int): Number of epochs to optimize the kernel at every .update() call.
            - `beta0` (float): Initial value of the beta parameter.
            - `pairing_strategy` (str): Strategy to pair the observations. Can be label-corresponding "label" or nearest-neighbour "nn". Defaults to "label".
            - `pairing_csv` (str): Path to a csv file containing the pairing of the dataset. If no file exists, the pairing will be performed and then stored at the specified location.
        
        Additional args:
            - `optimizer` (partially instantiated torch.optim.Optimizer): Custom optimizer to optimize the beta parameter. If None, the default optimizer is `torch.optim.Adam` with `lr = 0.1`.
                The optimizer must be partially instantiated using `functools.partial(torch.optim.Optimizer, **kwargs)` where only the `parameters` argument is left to be passed.
            - `processing_strategy` (str): Strategy to parallelize the optimization of the PA metric. Can be `"cpu"`, `"cuda"` or `"lightning"`. Defaults to `"cpu"`.
                . `"cpu"`: The metric will be computed in a single process locally.
                . `"cuda"`: The metric will be computed in parallel using the specified number of CUDA devices, either in an ongoing optimization or in a new process group. Override `_multiprocessing_conf()` if needed.
                . `"lightning"`: The metric is also adapted to work within the Pytorch Lightning framework. In this case the lightning `Trainer` will manage the device placement via `local_rank`.
            - `cuda_devices` (Union[List[str], int]): List of CUDA devices to be used in parallel. It must be specified in both `"lightning"` and `"cuda"` strategies.
            - `batch_size` (int): Batch size for the observations dataloader. Defaults to 16.
            - `num_workers` (int): Number of workers for the observations dataloader. Defaults to 0.
        """
        self.processing_strategy = processing_strategy
        self.cuda_devices = cuda_devices
        
        super().__init__(*args, **kwargs)
        self.accuracymetric = False # to distinguish easily from accuracy metric

        # (Multi)processing configuration
        self._multiprocessing_conf()
        
        # We will store the logPA for each of the (0, i) environment pairs, where i = 1:num_envs-1 are validation environments
        self.len_envmetrics = self.num_envs - 1 if self.num_envs > 1 else 1

        # Pass a custom optimizer
        self.partial_optimizer = optimizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize logs to store a value for each PosteriorAgreement() call. This is only for the 0-1 environments.
        self.log_beta, self.log_logPA, self.log_AFR_true, self.log_AFR_pred, self.log_accuracy = [], [], [], [], []
        
    def _multiprocessing_conf(self):
        """
        Defines the processing variables for a given `processing_strategy`.
            - For the `"cpu"` strategy, the device list is ["cpu"] and the process group is not initialized setting `ddp_init=None`.
            - For the `"cuda"` strategy, the device list is ["cuda:0", "cuda:1", ...] and the initialization of the process group
                is tracked by setting `ddp_init=[False]*len(device_list)`. Then `ddp_init[rank]` will be set to `True` once the process `rank` is initialized.
            - For the `"lightning"` strategy, the device list is ["cuda"] or ["cpu"] and `ddp_init=None` as the process group is initialized by the `Trainer`.
        """
        # Ensure that configuration makes sense
        assert self.processing_strategy in ["cpu", "cuda", "lightning"], "The processing strategy must be either 'cpu', 'cuda' or 'lightning'."
        
        num_cuda_dev = len(self.cuda_devices) if isinstance(self.cuda_devices, list) else self.cuda_devices
        if self.processing_strategy in ["cuda", "lightning"]:
            assert num_cuda_dev != None, "The number of cuda devices must be specified when `processing_strategy` is 'cuda' or 'lightning'." 
        
        if self.processing_strategy == "cuda":
            assert num_cuda_dev > 0, "The number of cuda devices must be greater than 0."
            
            if not torch.cuda.is_available():
                print("\nThe processing strategy is not 'cpu' but the only available device is 'cpu'. The metric will be slower than expected.")
                self.processing_strategy = "cpu"

        
        # Initialize multiprocessing configuration: self.device_list, ddp_init
        if self.processing_strategy == "cpu":
            self.device_list = ["cpu"]
            self.ddp_init = None

        elif self.processing_strategy == "cuda":
            assert num_cuda_dev <= torch.cuda.device_count()
            if dist.is_initialized():
                self.device_list = [f"cuda:{i}" for i in range(dist.get_world_size())]
            else:
                self.device_list = [f"cuda:{i}" for i in range(num_cuda_dev)]

            self.ddp_init = [False]*len(self.device_list)

        else: # processing_strategy == "lightning"
            # Pytorch Lightning already takes care of device placement
            self.device_list = ["cuda" if torch.cuda.is_available() and num_cuda_dev > 0 else "cpu"] # "cuda" or "cpu" is enough
            self.ddp_init = None

    def _get_world_size(self):
        """
        When training with the `"lightning"` strategy, the world size must be specified during initialization with the `cuda_devices` parameter, given that
        each process may be invisible to the rest. Otherwise, the world size is the number of devices in `device_list.
        """

        if self.processing_strategy == "lightning":
            if "cuda" in self.device_list[0]:
                return len(self.cuda_devices) if isinstance(self.cuda_devices, list) else self.cuda_devices
            else: # cpu
                return 1
        else:
            return len(self.device_list)
    
    def _get_current_dev(self, rank: int):
        """
        When training with the `"lightning"` strategy, the device allocation is managed by the `Trainer`, so tensors only need to be sent either to "cpu" or "cuda".
        Otherwise tensors must be sent to the corresponding device in `device_list` using the `rank` index.
        """

        if self.processing_strategy == "lightning":
            return self.device_list[0] # "cuda" or "cpu", lightning allocation is straightforward
        else: 
            return self.device_list[rank]
    
    def _initialize_optimization(self, rank: int):
        """
        When using DDP, the kernel model must be wrapped within torch.nn.parallel.DistributedDataParallel.DDP and accessed as `kernel.module`.
        Then the `module()` method in the kernel allows us to use it in the same way as in the base metric.
        """

        dev = self._get_current_dev(rank)
        kernel = PosteriorAgreementKernel(beta0=self.beta0, preds_2_factor=self.preds_2_factor).to(dev)
        if "cuda" in dev and self.processing_strategy == "cuda":
            kernel = DDP(kernel, device_ids=[rank])
        optimizer = self.partial_optimizer([kernel.module.beta]) if self.partial_optimizer else torch.optim.Adam([kernel.module.beta], lr=0.1) # default
        return kernel, optimizer
    
    # def _initialize_classifiers(...):

    def _delete_classifiers(self):
        self.classifier.to("cpu") 
        del self.classifier
        if self.classifier_val:
            self.classifier_val.to("cpu")
            del self.classifier_val
    
        # Teardown strategy
        torch.cuda.empty_cache()
        gc.collect()

    def _get_logits_from_batch(self, rank: int, batch, env_logit1: Optional[str] = "1"):
        """
        Now we allow for multiple datasets, so the second environment must be specified.
        """
        dev = self._get_current_dev(rank)

        envs = list(batch.keys())
        logits0, y = batch[envs[0]][0], batch[envs[0]][1]
        logits1 = batch[str(env_logit1)][0]
        return logits0.to(dev), logits1.to(dev), y.to(dev)
    
    def pa_optimize(self, rank: int):
        """
        Same optimization strategy as in the basemetric, but now we will store the mean beta value across processes.
        """
        beta = super().pa_optimize(rank)

        # We will store the mean value across processes
        dev = self._get_current_dev(rank)
        if "cuda" in dev:# and self.processing_strategy == "cuda":
            beta = torch.tensor(beta).to(dev)
            if dist.is_initialized():
                dist.all_reduce(beta, op=dist.ReduceOp.SUM)
                beta = beta.item() / self._get_world_size()

        return beta
    
    def pa_evaluate(self, rank: int, beta: float, env: Union[str, int], is_first_epoch: Optional[bool] = False):
        """
        Performs an evaluation epoch for a fixed beta to find the PA value.
        """
        # similar to the basemetric -------...-----------------------------------------------------------
        self.kernel.module.reset()

        total_samples, correct, correct_pred, correct_true = 0, 0, 0, 0
        for bidx, batch in enumerate(self.pa_dataloader):
            logits0, logits1, y = self._get_logits_from_batch(rank, batch, env)

            # Compute accuracy metrics only once
            if is_first_epoch:
                y_pred = torch.argmax(logits0, 1) # env 1
                y_pred_adv = torch.argmax(logits1, 1) # env2 or y
                correct_pred += (y_pred_adv == y_pred).sum().item()
                correct_true += (y_pred_adv == y).sum().item()
                correct += (torch.cat([y_pred, y_pred_adv]) == torch.cat([y, y])).sum().item()
                total_samples += len(y)

            # Update logPA
            self.kernel.module.evaluate(logits0, logits1, beta)
            # ------------------------------------------------------------------------------------------

        dev = self._get_current_dev(rank)
        logPA = self.kernel.module.log_posterior().to(dev)

        # Compute accuracy metrics across devices
        if "cuda" in dev and dist.is_initialized():# and self.processing_strategy == "cuda":
            dist.all_reduce(logPA, op=dist.ReduceOp.SUM) # add the logPA from all devices
            if is_first_epoch:
                dist.all_reduce(torch.tensor(total_samples).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(total_samples).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct_pred).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct_true).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct).to(dev), op=dist.ReduceOp.SUM)

        return {
            "logPA": logPA.item(),
            "AFR_pred": correct_pred/total_samples,
            "AFR_true": correct_true/total_samples,
            "accuracy": correct/(2*total_samples)
        } if is_first_epoch else {"logPA": logPA.item()}


    def initialize_distributed(self, rank: int, destroy_process_group: Optional[bool] = False):
        """
        Calls `.pa_update()` through the corresponding subprocess. Only called if `self.processing_strategy == 'cuda'`.
        """
        # Initialize the process only once, even if the .update() is called several times during a training procedure.
        if self.ddp_init[rank] == False: 
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ["MASTER_PORT"] = "50000"
            init_process_group(backend="nccl", rank=rank, world_size=self._get_world_size())
            torch.cuda.set_device(rank)
            self.ddp_init[rank] = True

        # Call PA update function
        self.pa_update(rank)

        torch.cuda.synchronize()
        gc.collect()

        dist.barrier()
        if destroy_process_group and dist.is_initialized():
            dist.destroy_process_group()

    def _compute_logits_dataset(self, rank: int):
        """
        Computes a LogitsDataset at the beginning of the `.update()` call so that the following optimization epochs
        can be faster and more efficient. Each LogitsDataset contains an evaluated subset of the images, depending on the
        number of processes launched.
        """
        dev = self._get_current_dev(rank)
        world_size = self._get_world_size()

        # Move classifiers to the appropiate device
        self.classifier.to(dev)
        if self.classifier_val:
            self.classifier_val.to(dev)
            
        dataloader = DataLoader(
            dataset=self.dataset,
            collate_fn=MultiEnv_collate_fn,

            batch_size = self.batch_size,
            num_workers = self.num_workers, 

            pin_memory = ('cuda' in self.device_list[0]),
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, 
                drop_last=False,
                shuffle=False,
                num_replicas=world_size,
                rank=rank
            )
        )

        with torch.no_grad():
            y_totensor = [None]*len(dataloader)
            X_totensor = [None]*len(dataloader)
            for bidx, batch in enumerate(dataloader):
                if bidx == 0: # initialize logits dataset
                    envs = list(batch.keys())
                    if len(envs) != self.num_envs:
                        raise ValueError("There is a problem with the configuration of the Dataset and/or the DataLoader collate function.")
                
                X_list = [batch[envs[e]][0] for e in range(self.num_envs)]
                Y_list = [batch[envs[e]][1] for e in range(self.num_envs)]
                if self.pairing_strategy == "label" and not all([torch.equal(Y_list[0], Y_list[i]) for i in range(1, len(Y_list))]): # all labels must be equal
                    print("The labels of the two environments are not the same.")
                    # raise ValueError("The labels of the two environments must be the same.")
                
                y_totensor[bidx] = Y_list[0] # same for all environments
                if self.classifier_val: # then the validation with additional datasets uses the second classifier
                    X_totensor[bidx] = [self.classifier(X_list[0].to(dev))] + [self.classifier_val(X_list[i].to(dev)) for i in range(1, len(X_list))]
                    
                else: # subset has two elements, each with the same labels
                    X_totensor[bidx] = [self.classifier(X.to(dev)) for X in X_list]

            logits_list = [torch.cat([X_totensor[i][e] for i in range(len(y_totensor))]) for e in range(self.num_envs)]
            y = torch.cat(y_totensor)

            # Remove classifiers from memory
            self._delete_classifiers()
            return LogitsDataset(logits_list, y)

    def pa_update(self, rank: int):
        logits_dataset = self._compute_logits_dataset(rank)
        logits_batch_size = adjust_batch_size(
            self.batch_size,
            self.dataset[0][list(self.dataset[0].keys())[0]][0],
            logits_dataset[0][list(logits_dataset[0].keys())[0]][0]
        )
        logits_batch_size = min(
            logits_batch_size,
            len(logits_dataset)
        )

        # logits_batch_size = self.batch_size # logits_batch_size, #TODO: Change after tests
        self.pa_dataloader = DataLoader(
                        dataset=logits_dataset,
                        batch_size=logits_batch_size,
                        num_workers=0, # we won't create subprocesses inside a subprocess, and data is very light
                        pin_memory=False, # only dense CPU tensors can be pinned

                        drop_last = False,
                        sampler = RandomSampler(logits_dataset)
        )

        # Initialize kernel and optimizer every time
        self.kernel, self.optimizer = self._initialize_optimization(rank)

        # Optimize beta for every batch within an epoch, for every epoch
        for epoch in range(self.pa_epochs):
            self.betas[epoch] = self.pa_optimize(rank)

            iterated_envs = list(range(1, self.num_envs)) if self.num_envs > 1 else [0]
            for i, it_env in enumerate(iterated_envs):
                metric_dict = self.pa_evaluate(rank, self.betas[epoch].item(), it_env, is_first_epoch = (epoch == 0))
                self.logPAs[i, epoch] = metric_dict["logPA"]

                # Accuracy metrics are the same for every epoch
                if epoch == 0:
                    self.afr_pred[i] = metric_dict["AFR_pred"]
                    self.afr_true[i] = metric_dict["AFR_true"]
                    self.accuracy[i] = metric_dict["accuracy"]

    def update(
        self, 
        classifier: torch.nn.Module, 
        classifier_val: Optional[torch.nn.Module] = None, 
        destroy_process_group: Optional[bool] = False,
        local_rank: Optional[int] = None
    ):
        
        if self.processing_strategy == "lightning":
            assert local_rank != None, "The local rank must be passed if the strategy is 'lightning'."

        # Initialize betas for optimization (env0 vs env1) for every epoch
        self.betas = torch.zeros(self.pa_epochs, dtype=torch.float64).to("cpu")

        # LogPAs must be stored for every epoch, and also for every validation environment.
        self.logPAs = torch.full((self.len_envmetrics, self.pa_epochs), -float('inf'), dtype=torch.float64)

        # Initialize accuracy metrics for all the environments
        self.afr_pred = torch.zeros(self.len_envmetrics, dtype=torch.float).to("cpu")
        self.afr_true = torch.zeros(self.len_envmetrics, dtype=torch.float).to("cpu")
        self.accuracy = torch.zeros(self.len_envmetrics, dtype=torch.float).to("cpu")

        self.classifier, self.classifier_val = self._initialize_classifiers(classifier, classifier_val)

        # Optimize beta depending on the strategy and the devices available
        if dist.is_initialized(): # ongoing cuda or ddp lightning
            self.pa_update(dist.get_rank() if self.processing_strategy == "cuda" else local_rank)
        else:
            if self.processing_strategy == "cuda" and "cuda" in self.device_list[0]:
                mp.spawn(
                    self.initialize_distributed,
                    args=(destroy_process_group,),
                    nprocs=self._get_world_size(),
                    join=True
                )

            else: # "cpu", either lightning or not
                self.pa_update(0)

    def compute(self):
        # Maximum logPA for the first environment pair (env0 vs env1)
        self.selected_index = torch.argmax(self.logPAs[0, :]).item()

        metrics_dict = {
            i: {
                "beta": self.betas[self.selected_index].item(),
                "logPA": self.logPAs[i, self.selected_index].item(),
                "AFR_pred": self.afr_pred[i].item(),
                "AFR_true": self.afr_true[i].item(),
                "acc_pa": self.accuracy[i].item()
            }
            for i in range(self.len_envmetrics)
        }

        # Store values in the log for the first environment pair
        self.log_beta.append(metrics_dict[0]["beta"])
        self.log_logPA.append(metrics_dict[0]["logPA"])
        self.log_AFR_true.append(metrics_dict[0]["AFR_true"])
        self.log_AFR_pred.append(metrics_dict[0]["AFR_pred"])
        self.log_accuracy.append(metrics_dict[0]["acc_pa"])

        return metrics_dict


    