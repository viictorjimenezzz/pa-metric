"""
This test will check that the PA metric provides the same results that the existing PA module when the data is controlled.
"""

import hydra
from omegaconf import DictConfig

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from pytorch_lightning import seed_everything, LightningModule, LightningDataModule

from .utils import get_pa_metrics, get_acc_metrics
from pametric.datautils import LogitsDataset, MultienvDataset
from pametric.metrics import PosteriorAccuracy, PosteriorAgreement

from copy import deepcopy

def test_basemetric(cfg: DictConfig):
    """
    This test checks that the optimization over beta yields the same results in the PA module
    and the basemetric.
    """
    seed_everything(42, workers=True)

    ## DATA INITIALIZATION:
    # Main datamodule:
    datamodule_main: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.main)
    datamodule_main.prepare_data()
    datamodule_main.setup("fit")

    # Impose sequential sampler
    main_traindl = datamodule_main.train_dataloader()
    main_traindl = DataLoader(
        main_traindl.dataset, 
        batch_size=main_traindl.batch_size, 
        sampler=SequentialSampler(main_traindl.dataset), 
        num_workers=main_traindl.num_workers, 
        collate_fn=main_traindl.collate_fn
    )

    # Logits datamodule
    datamodule_palogs: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.logits)
    datamodule_palogs.prepare_data()
    datamodule_palogs.setup("fit")

    # Impose sequential sampler
    palogs_traindl = datamodule_palogs.train_dataloader()
    palogs_traindl = DataLoader(
        palogs_traindl.dataset, 
        batch_size=palogs_traindl.batch_size, 
        sampler=SequentialSampler(palogs_traindl.dataset), 
        num_workers=palogs_traindl.num_workers, 
        collate_fn=palogs_traindl.collate_fn
    )

    # USING THE PA MODULE
    pamodule_palogs: LightningModule = hydra.utils.instantiate(cfg.pa_metric.pa_module)
    trainer = hydra.utils.instantiate(cfg.pa_metric.trainer.cpu)
    trainer.fit(
            model=pamodule_palogs, 
            # Same data for training and validation, as in the metric.
            train_dataloaders=palogs_traindl, 
            val_dataloaders=palogs_traindl
    )
    acc_metrics = get_acc_metrics(pamodule_palogs)
    AFR_pred = acc_metrics[0]
    AFR_true = acc_metrics[1]
    acc_pa = acc_metrics[2]

    beta_epoch_palogs, logPA_epoch_palogs = get_pa_metrics(pamodule_palogs)
    assert len(beta_epoch_palogs) == cfg.pa_metric.trainer.cpu.max_epochs, "Some beta values are not being stored properly."
    assert len(logPA_epoch_palogs) == cfg.pa_metric.trainer.cpu.max_epochs, "Some logPA values are not being stored properly."

    ## USING THE PA BASEMETRIC
    partial_basemetric = hydra.utils.instantiate(cfg.pa_metric.pa_basemetric)
    pabasemetric = partial_basemetric(main_traindl.dataset)
    pabasemetric.update(
        hydra.utils.instantiate(cfg.pa_metric.datamodules.logits.classifier)
    )

    """
    Since palogs_traindl and palogs_valdl are the same, the base metric should give the same results as the PA module.
    """
    assert torch.equal(beta_epoch_palogs, pabasemetric.betas), "The beta values do not coincide."
    assert torch.equal(logPA_epoch_palogs, pabasemetric.logPAs), "The logPA values do not coincide."
    assert AFR_true == pabasemetric.afr_true, "The AFR_true values do not coincide."
    assert AFR_pred == pabasemetric.afr_pred, "The AFR_pred values do not coincide."
    assert acc_pa == pabasemetric.accuracy, "The accuracy values do not coincide."

    """
    This would be the dictionary we would be getting if we called simply basemetric(logits), instead of basemetric.update(logits).
    This is how the metric will be called in any implementation.
    """
    results_dict = pabasemetric.compute()
    print("\nResults from the base metric:")
    print(results_dict)

    assert results_dict["logPA"] == max(pabasemetric.logPAs), "The logPA value is not the maximum of the logPAs."

    print("\nTest passed.")


def test_pametric_cpu(cfg: DictConfig): 
    """
    Check that the results of the PA metric are the same as the PA base metric when using the CPU. The basemetric has already
    been tested against the PA module, and passing the test would mean that the PA metric is also working properly in the CPU.
    """    

    # INITIALIZE IMAGES DATALOADER
    datamodule_images: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.images)
    datamodule_images.prepare_data()
    datamodule_images.setup("fit")

    # I disable shuffling so that the results can be compared
    images_traindl = datamodule_images.train_dataloader()
    images_traindl = DataLoader(
        images_traindl.dataset, 
        batch_size=images_traindl.batch_size, 
        sampler=SequentialSampler(images_traindl.dataset), 
        num_workers=images_traindl.num_workers, 
        collate_fn=images_traindl.collate_fn,
        drop_last=False
    )

    # Using the PA basemetric  ---------------------------------------------------------------
    pabasemetric_partial = hydra.utils.instantiate(cfg.pa_metric.metrics.basemetric)
    pabasemetric = pabasemetric_partial(dataset = images_traindl.dataset)
    pabasemetric.update(hydra.utils.instantiate(cfg.pa_metric.classifier))

    # Using the PosteriorAgreement metric ----------------------------------------------------
    pa_metric_partial = hydra.utils.instantiate(cfg.pa_metric.metrics.fullmetric) # initialize the metric for CPU
    pa_metric = pa_metric_partial(dataset = images_traindl.dataset)
    pa_metric.update(hydra.utils.instantiate(cfg.pa_metric.classifier))

    assert torch.equal(pa_metric.betas, pabasemetric.betas), "The beta values do not coincide with the logits results."
    assert torch.equal(pa_metric.logPAs[0, :], pabasemetric.logPAs), "The logPA values do not coincide with the logits results."
    assert pa_metric.afr_true[0].item() == pabasemetric.afr_true, "The AFR_true values do not coincide with the logits results."
    assert pa_metric.afr_pred[0].item() == pabasemetric.afr_pred, "The AFR_pred values do not coincide with the logits results."
    assert pa_metric.accuracy[0].item() == pabasemetric.accuracy, "The accuracy values do not coincide with the logits results."

    print("\nTest passed.")

import time
def test_pametric_ddp(cfg: DictConfig):
    """
    Check that the results of the PA metric are the same regardless of the data partitioning strategy and the
    number of GPUs used. This is important as the PA metric bears a custom DDP implementation.
    """

    # We will only work with the logits dataloader, as it makes it easier.
    datamodule_main: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.images)
    datamodule_main.prepare_data()
    datamodule_main.setup("fit")

    # I disable shuffling so that the results can be compared
    main_traindl = datamodule_main.train_dataloader()
    main_traindl = DataLoader(
        main_traindl.dataset, 
        batch_size=main_traindl.batch_size, 
        sampler=SequentialSampler(main_traindl.dataset), 
        num_workers=main_traindl.num_workers, 
        collate_fn=main_traindl.collate_fn,
        drop_last=False
    )

    # Initialization of the metric
    pa_ds = main_traindl.dataset # get the dataset
    pa_metric_partial = hydra.utils.instantiate(cfg.pa_metric.metric)

    # Metric running in the CPU vs CUDA ---------------------------------------------------------------
    pa_metric_cpu = pa_metric_partial(
        dataset = pa_ds,
        processing_strategy = "cpu"
    )
    pa_metric_cpu.update(hydra.utils.instantiate(cfg.pa_metric.classifier))

    pa_metric_cuda = pa_metric_partial(
        dataset = pa_ds,
        processing_strategy = "cuda",
        cuda_devices = 4
    )
    pa_metric_cuda.update(hydra.utils.instantiate(cfg.pa_metric.classifier), destroy_process_group=True) # because it's the last time it will be called

    assert torch.allclose(pa_metric_cpu.betas, pa_metric_cuda.betas), "The beta values do not coincide."
    assert torch.allclose(pa_metric_cpu.logPAs, pa_metric_cuda.logPAs), "The logPA values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_true, pa_metric_cuda.afr_true), "The AFR_true values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_pred, pa_metric_cuda.afr_pred), "The AFR_pred values do not coincide."
    assert torch.allclose(pa_metric_cpu.accuracy, pa_metric_cuda.accuracy), "The accuracy values do not coincide."

    # Metric running in less number of GPUs than available -----------------------------------------
    if dist.is_initialized():
        dist.destroy_process_group()
    pa_metric_cuda2 = pa_metric_partial(
        dataset = pa_ds,
        processing_strategy = "cuda",
        cuda_devices = 2 # change number of devices
    )
    pa_metric_cuda2.update(hydra.utils.instantiate(cfg.pa_metric.classifier), destroy_process_group=True)
    
    assert torch.allclose(pa_metric_cuda.betas, pa_metric_cuda2.betas), "The beta values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.logPAs[0], pa_metric_cuda2.logPAs[0]), "The logPA values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.afr_true, pa_metric_cuda2.afr_true), "The AFR_true values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.afr_pred, pa_metric_cuda2.afr_pred), "The AFR_pred values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.accuracy, pa_metric_cuda2.accuracy), "The accuracy values do not coincide with different number of GPUs."

    # Metric performing several calls in a row with the same data -----------------------------------
    if dist.is_initialized():
        dist.destroy_process_group()
    pa_metric_cuda = pa_metric_partial(
        dataset = pa_ds,
        cuda_devices = 4,
        processing_strategy = "cuda")
    pa_metric_cuda(hydra.utils.instantiate(cfg.pa_metric.classifier), destroy_process_group=False)
    pa_metric_cuda(hydra.utils.instantiate(cfg.pa_metric.classifier), destroy_process_group=False)
    pa_metric_cuda(hydra.utils.instantiate(cfg.pa_metric.classifier), destroy_process_group=True)

    print("\nThe log of results for multiple calls (3) is working: ")
    print("Logged logPA: ", pa_metric_cuda.log_logPA)
    print("Logged beta: ", pa_metric_cuda.log_beta)
    print("Logged AFR_true: ", pa_metric_cuda.log_AFR_true)
    print("Logged AFR_pred: ", pa_metric_cuda.log_AFR_pred)
    print("Logged accuracy: ", pa_metric_cuda.log_accuracy)

    assert torch.allclose(pa_metric_cpu.betas, pa_metric_cuda.betas), "The beta values do not coincide."
    assert torch.allclose(pa_metric_cpu.logPAs[0], pa_metric_cuda.logPAs[0]), "The logPA values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_true, pa_metric_cuda.afr_true), "The AFR_true values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_pred, pa_metric_cuda.afr_pred), "The AFR_pred values do not coincide."
    assert torch.allclose(pa_metric_cpu.accuracy, pa_metric_cuda.accuracy), "The accuracy values do not coincide."

    # Performing many epochs ---------------------------------------------------------------------
    start = time.time()
    if dist.is_initialized():
        dist.destroy_process_group()
    pa_metric_cuda_long = pa_metric_partial(
        dataset = pa_ds,
        pa_epochs = 1000,
        cuda_devices = 4,
        processing_strategy = "cuda")
    metric_dict_long = pa_metric_cuda_long(destroy_process_group=True)
    print("\nTime for 1000 epochs: ", time.time() - start)

    print("\nTest passed.")


def test_pametric_logits(cfg: DictConfig):
    """
    We will test if the logits generated within the PosteriorAgreement metric in DDP mode are the same
    as the ones provided by the LogitsDatamodule.
    """

    # Non-paired dataset of images:
    datamodule_main: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.main)
    datamodule_main.prepare_data()
    datamodule_main.setup("fit")
    main_traindl = datamodule_main.train_dataloader()
    main_dataset = main_traindl.dataset

    # Paired dataset of logits:
    datamodule_palogs: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodules.pa_logits)
    datamodule_palogs.prepare_data()
    datamodule_palogs.setup("fit")
    palogs_traindl = datamodule_palogs.train_dataloader()
    palogs_dataset = palogs_traindl.dataset
    assert palogs_dataset.__class__.__name__ == "LogitsDataset", "The PA_logits dataset is not being properly loaded."

    pa_metric_partial = hydra.utils.instantiate(cfg.pa_metric.metric)
    pa_metric_cpu = pa_metric_partial(
                    dataset = main_dataset, 
                    processing_strategy = "cpu"
    )

    classifier = hydra.utils.instantiate(cfg.pa_metric.datamodules.pa_logits.classifier)
    pa_metric_cpu.classifier, pa_metric_cpu.classifier_val = pa_metric_cpu._initialize_classifiers(classifier)

    # Initialize classifier: Same as the one used in the PA logits.
    cpu_dataset = pa_metric_cpu._compute_logits_dataset(0)
    assert cpu_dataset.__class__.__name__ == "LogitsDataset", "The logits dataset is not being properly loaded from the metric."

    """
    The LogitsDataset provided by the method in the PosteriorAgreement metric should be the same as the one
    generated from the same dataset in the PA_logits datamodule.
    """
    assert palogs_dataset.num_envs == cpu_dataset.num_envs , "The number of environments does not coincide."
    assert torch.equal(palogs_dataset.y, cpu_dataset.y), "The labels are not the same."
    for e in range(palogs_dataset.num_envs):
        assert torch.allclose(palogs_dataset.logits[e], cpu_dataset.logits[e]), f"The logits in environment {e} are not the same."

    print("\nCPU test passed.\n")
    exit()

    """
    Now we must check that the PA optimization results are the same in the CPU and in DDP mode using the same dataset.

    IMPORTANT: We must instantiate the dataset again, as it has already been paired. A second pairing would mess it up.
    """
    pa_metric_cpu.update(classifier)
    pa_metric_cuda = pa_metric_partial(
                    dataset = main_dataset, 
                    cuda_devices = 4,
                    processing_strategy = "cuda"
    )
    pa_metric_cuda.update(classifier, destroy_process_group=True)

    assert torch.allclose(pa_metric_cpu.betas, pa_metric_cuda.betas), "The beta values do not coincide."
    assert torch.allclose(pa_metric_cpu.logPAs, pa_metric_cuda.logPAs), "The logPA values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_true, pa_metric_cuda.afr_true), "The AFR_true values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_pred, pa_metric_cuda.afr_pred), "The AFR_pred values do not coincide."
    assert torch.allclose(pa_metric_cpu.accuracy, pa_metric_cuda.accuracy), "The accuracy values do not coincide."
        
    print("\nTest passed.")


def test_accuracymetric(cfg):
    """
    Test the subclass of posterior accuracy, where one of the distributions is fixed.
    """

    # We will only work with the logits dataloader, as it makes it easier.
    datamodule_main: LightningDataModule = hydra.utils.instantiate(cfg.pa_metric.datamodule)
    datamodule_main.prepare_data()
    datamodule_main.setup("fit")

    # I select the first dataset from the MultienvDataset, as the Accuracy metric only requires one.
    ds = MultienvDataset([datamodule_main.train_ds.dset_list[0]])

    # Initialization of the metric
    pa_metric_partial = hydra.utils.instantiate(cfg.pa_metric.metric)

    # Initialization of the fixed classifier used to update the metric
    classifier = hydra.utils.instantiate(cfg.pa_metric.classifier)

    # # Metric running in the CPU vs CUDA ---------------------------------------------------------------
    pa_metric_cpu = pa_metric_partial(
        dataset = ds,
        sharpness_factor=1.5,
        processing_strategy = "cpu")
    pa_metric_cpu.update(classifier)

    pa_metric_cuda = pa_metric_partial(
        dataset = ds,
        sharpness_factor=1.5,
        cuda_devices = 4,
        processing_strategy = "cuda")
    pa_metric_cuda.update(classifier, destroy_process_group=True) # because it's the last time it will be called

    assert torch.allclose(pa_metric_cpu.betas, pa_metric_cuda.betas), "The beta values do not coincide."
    assert torch.allclose(pa_metric_cpu.logPAs, pa_metric_cuda.logPAs), "The logPA values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_true, pa_metric_cuda.afr_true), "The AFR_true values do not coincide."
    assert torch.allclose(pa_metric_cpu.afr_pred, pa_metric_cuda.afr_pred), "The AFR_pred values do not coincide."
    assert torch.allclose(pa_metric_cpu.accuracy, pa_metric_cuda.accuracy), "The accuracy values do not coincide."

    # Metric running in less number of GPUs than available -----------------------------------------
    pa_metric_cuda2 = pa_metric_partial(
        dataset = ds,
        sharpness_factor=1.5,
        cuda_devices = 2, # change number of devices
        processing_strategy = "cuda")
    pa_metric_cuda2.update(classifier, destroy_process_group=True)

    assert torch.allclose(pa_metric_cuda.betas, pa_metric_cuda2.betas), "The beta values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.logPAs, pa_metric_cuda2.logPAs), "The logPA values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.afr_true, pa_metric_cuda2.afr_true), "The AFR_true values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.afr_pred, pa_metric_cuda2.afr_pred), "The AFR_pred values do not coincide with different number of GPUs."
    assert torch.allclose(pa_metric_cuda.accuracy, pa_metric_cuda2.accuracy), "The accuracy values do not coincide with different number of GPUs."

    # Metric performing several calls woth different sharpness -----------------------------------
    for sharpness in [1.001, 1.01, 1.1, 1.5, 2, 10, 50]:
        pa_accuracymetric = pa_metric_partial(
            dataset = ds,
            sharpness_factor=sharpness,
            cuda_devices = 0,
            processing_strategy = "cpu")
        pa_accuracymetric(classifier)

        pa_accuracymetric2 = pa_metric_partial(
            dataset = ds,
            sharpness_factor=sharpness,
            cuda_devices = 4,
            processing_strategy = "cuda")
        pa_accuracymetric2(classifier)

        print("\nFor sharpness factor: ", sharpness)
        print("beta: ", pa_accuracymetric.betas, pa_accuracymetric2.betas)
        print("logPA: ", pa_accuracymetric.logPAs, pa_accuracymetric2.logPAs)

    # Performing many epochs ---------------------------------------------------------------------
    start = time.time()
    pa_metric_cuda_long = pa_metric_partial(
        dataset = ds,
        sharpness_factor=1.5,
        pa_epochs = 1000,
        cuda_devices = 4,
        processing_strategy = "cuda")
    metric_dict_long = pa_metric_cuda_long(classifier, destroy_process_group=True)
    print("\nTime for 1000 epochs: ", time.time() - start)

    print("\nTest passed.")






