"""
This test will check if the PosteriorAgreement metric works properly when processing_strategy == "lightning"; that is, when
the metric is called within an ongoing parallelized process. 

The results provided the metric in the Callback should be the same as the ones provided by the metric within the
LightningModule, and the results for the last epoch should as well coincide with those obtained with the PA module.
"""

import hydra
from omegaconf import DictConfig

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from pytorch_lightning import LightningModule, LightningDataModule

from .utils import get_acc_metrics, get_pa_metrics
from copy import deepcopy

from torch import nn, argmax, optim
from pametric.lightning.metric_callback import PA_Callback
from pametric.pairing import PosteriorAgreementDatasetPairing
from pametric.metrics import PosteriorAgreement

class Vanilla(LightningModule):
    def __init__(
        self,
        num_classes: int,
        classifier: nn.Module,
        metric,
        log_every_n_epochs:int
    ):
        super().__init__()

        self.model = deepcopy(classifier).eval()
        self.model_to_train = classifier.train()
        self.loss = nn.CrossEntropyLoss()
        self.n_classes = int(num_classes)

        self.metric = metric
        self.log_every_n_epochs = log_every_n_epochs

    def training_step(self, batch, batch_idx):
        # Adapt to Multienv_collate_fn so that I dont have to instantiate main datamodule twice
        env_names = list(batch.keys())
        x = torch.cat([batch[env][0] for env in env_names])
        y = torch.cat([batch[env][1] for env in env_names])

        with torch.set_grad_enabled(True):
            logits = self.model_to_train(x)
            loss = self.loss(input=logits, target=y)
        assert loss.requires_grad
        return {"loss": loss}
    
    def on_train_epoch_start(self):
        # If we already computed PA in the previous iteration, then we can compare the results:
        if self.current_epoch + 1 > self.log_every_n_epochs:
            print(f"We are checking the results at epoch {self.current_epoch}")
            assert torch.allclose(self.metric.betas, self.trainer.callbacks[0].pa_metric.betas)
            assert torch.allclose(self.metric.logPAs[0], self.trainer.callbacks[0].pa_metric.logPAs[0])
            assert torch.allclose(self.metric.afr_pred, self.trainer.callbacks[0].pa_metric.afr_pred)
            assert torch.allclose(self.metric.afr_true, self.trainer.callbacks[0].pa_metric.afr_true)
            assert torch.allclose(self.metric.accuracy, self.trainer.callbacks[0].pa_metric.accuracy)
    
    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.log_every_n_epochs == 0:
            print(f"\nWE ARE COMPUTING THE METRIC at epoch {self.current_epoch}\n")     
            self.metric.update(
                deepcopy(self.model).eval(), 
                local_rank=self.trainer.local_rank,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return {"optimizer": optimizer}
    
    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
            **kwargs,
        ):
        """
        We will set gradients to zero after the optimizer step, so that the model is not optimized.
        This is done so that I can pass an already trained classifier and obtain the same results for the
        PA over and over again.
        """
        
        # call the 'backward' function on the current loss
        optimizer_closure()

        # Set gradients to zero to avoid the model from being optimized
        for param in self.parameters():
            if param.grad is not None:
                param.grad *= 0  # This zeros the gradients
        optimizer.step()
    
"""
- Only in DDP as it's where it makes sense to test a model.
- Train Vanilla with the callback in the Trainer and with the metric inside. The result should be the same.
- The metric outside and the PA_module outside should also give the same result.
"""

def test_pa_callback(cfg: DictConfig):

    # Main DataModule -----------------------------------------------------------------------
    # This is the datamodule that will train the model
    datamodule_main: LightningDataModule = hydra.utils.instantiate(cfg.pa_callback.datamodules.main)
    datamodule_main.prepare_data()
    datamodule_main.setup("fit")

    # PA DataModule -------------------------------------------------------------------------
    datamodule_pa: LightningDataModule = hydra.utils.instantiate(cfg.pa_callback.datamodules.pa)
    datamodule_pa.prepare_data()
    datamodule_pa.setup("fit")

    # I disable shuffling so that the results can be compared
    pa_traindl = datamodule_pa.train_dataloader()
    pa_traindl = DataLoader(
        pa_traindl.dataset, 
        batch_size=pa_traindl.batch_size, 
        sampler=SequentialSampler(pa_traindl.dataset), 
        num_workers=0,#pa_traindl.num_workers, 
        collate_fn=pa_traindl.collate_fn,
        drop_last=False
    )

    #______________________________________ CPU ______________________________________

    # Instantiation of the callback
    pa_callback_partial = hydra.utils.instantiate(cfg.pa_callback.pa_callback)
    pa_callback = pa_callback_partial(
        dataset = pa_traindl.dataset,
        cuda_devices = 0
    )

    # Instantiation of the metric
    pa_metric_partial = hydra.utils.instantiate(cfg.pa_callback.pa_metric)
    pa_metric = pa_metric_partial(
                    dataset = pa_traindl.dataset,
                    processing_strategy = "lightning",
                    cuda_devices=0
    )

    # We train the model with the callback and the metric inside:
    trainer_vanilla_partial = hydra.utils.instantiate(cfg.pa_callback.trainers.vanilla.cpu)
    trainer_vanilla = trainer_vanilla_partial(
        callbacks=[pa_callback]
    )
    vanilla_model_partial = hydra.utils.instantiate(cfg.pa_callback.vanilla_model)
    vanilla_model = vanilla_model_partial(
        metric = pa_metric
    )
    trainer_vanilla.fit(vanilla_model, datamodule_main)

    """
    The overall results of the callback and the metric should be exactly the same.
    """

    # Check PA results from the last epoch it was computed. For other epochs it has been checked from within the LightnignModule
    assert torch.allclose(vanilla_model.metric.betas, vanilla_model.trainer.callbacks[0].pa_metric.betas)
    assert torch.allclose(vanilla_model.metric.logPAs, vanilla_model.trainer.callbacks[0].pa_metric.logPAs)
    assert torch.allclose(vanilla_model.metric.afr_pred, vanilla_model.trainer.callbacks[0].pa_metric.afr_pred)
    assert torch.allclose(vanilla_model.metric.afr_true, vanilla_model.trainer.callbacks[0].pa_metric.afr_true)
    assert torch.allclose(vanilla_model.metric.accuracy, vanilla_model.trainer.callbacks[0].pa_metric.accuracy)

    # Check that the values are different
    assert not torch.equal(vanilla_model.metric.betas, vanilla_model.metric.betas[torch.randperm(len(vanilla_model.metric.betas))])

    # We also check the final results of the whole PA optimization (i.e. the PA selected for each call)    
    assert torch.allclose(torch.tensor(vanilla_model.metric.log_beta), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_beta))
    assert torch.allclose(torch.tensor(vanilla_model.metric.log_logPA), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_logPA))
    assert torch.allclose(torch.tensor(vanilla_model.metric.log_AFR_pred), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_AFR_pred))
    assert torch.allclose(torch.tensor(vanilla_model.metric.log_AFR_pred), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_AFR_pred))
    assert torch.allclose(torch.tensor(vanilla_model.metric.log_accuracy), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_accuracy))
    
    # """
    # Now we will check that the same results are obtained when using the MAIN datamodule also for instantiating the callback, etc.
    # """
    # main_ds = PosteriorAgreementDatasetPairing(datamodule_main.train_ds)

    # # Instantiation of the callback
    # pa_callback_partial = hydra.utils.instantiate(cfg.pa_callback.pa_callback)
    # pa_callback = pa_callback_partial(
    #     dataset = main_ds,
    #     cuda_devices = 0
    # )

    # # Instantiation of the metric
    # pa_metric_partial = hydra.utils.instantiate(cfg.pa_callback.pa_metric)
    # pa_metric = pa_metric_partial(
    #                 dataset = main_ds,
    #                 processing_strategy = "lightning",
    #                 cuda_devices=0
    # )

    # # We train the model with the callback and the metric inside:
    # trainer_vanilla_partial = hydra.utils.instantiate(cfg.pa_callback.trainers.vanilla.cpu)
    # trainer_vanilla = trainer_vanilla_partial(
    #     callbacks=[pa_callback]
    # )
    # vanilla_model_partial = hydra.utils.instantiate(cfg.pa_callback.vanilla_model)
    # vanilla_model2 = vanilla_model_partial(
    #     metric = pa_metric
    # )
    # trainer_vanilla.fit(vanilla_model2, datamodule_main)

    # # First we check that the results are the same wrt the previous implementation.
    # assert vanilla_model.metric.afr_pred.item() == vanilla_model2.metric.afr_pred.item(), "The AFR_pred is not the same."
    # assert vanilla_model.metric.afr_true.item() == vanilla_model2.metric.afr_true.item(), "The AFR_true is not the same."
    # assert vanilla_model.metric.accuracy.item() == vanilla_model2.metric.accuracy.item(), "The accuracy is not the same."

    # # Check PA results from the last epoch it was computed. For other epochs it has been checked from within the LightnignModule
    # assert torch.allclose(vanilla_model2.metric.betas, vanilla_model2.trainer.callbacks[0].pa_metric.betas)
    # assert torch.allclose(vanilla_model2.metric.logPAs, vanilla_model2.trainer.callbacks[0].pa_metric.logPAs)
    # assert torch.allclose(vanilla_model2.metric.afr_pred, vanilla_model2.trainer.callbacks[0].pa_metric.afr_pred)
    # assert torch.allclose(vanilla_model2.metric.afr_true, vanilla_model2.trainer.callbacks[0].pa_metric.afr_true)
    # assert torch.allclose(vanilla_model2.metric.accuracy, vanilla_model2.trainer.callbacks[0].pa_metric.accuracy)

    # # Check that the values are different
    # assert not torch.equal(vanilla_model2.metric.betas, vanilla_model2.metric.betas[torch.randperm(len(vanilla_model2.metric.betas))])

    # # We also check the final results of the whole PA optimization (i.e. the PA selected for each call)    
    # assert torch.allclose(torch.tensor(vanilla_model2.metric.log_beta), torch.tensor(vanilla_model2.trainer.callbacks[0].pa_metric.log_beta))
    # assert torch.allclose(torch.tensor(vanilla_model2.metric.log_logPA), torch.tensor(vanilla_model2.trainer.callbacks[0].pa_metric.log_logPA))
    # assert torch.allclose(torch.tensor(vanilla_model2.metric.log_AFR_pred), torch.tensor(vanilla_model2.trainer.callbacks[0].pa_metric.log_AFR_pred))
    # assert torch.allclose(torch.tensor(vanilla_model2.metric.log_AFR_pred), torch.tensor(vanilla_model2.trainer.callbacks[0].pa_metric.log_AFR_pred))
    # assert torch.allclose(torch.tensor(vanilla_model2.metric.log_accuracy), torch.tensor(vanilla_model2.trainer.callbacks[0].pa_metric.log_accuracy))
    
    # """
    # Now we will call the metric from outside again, and the PA module. 
    # """
    
    # # Results should be the same (comparison within the model)
    # pa_metric = pa_metric_partial(
    #                 dataset = pa_traindl.dataset,
    #                 processing_strategy = "cpu"
    # )
    # pa_metric.update(deepcopy(vanilla_model.model).eval())

    # pa_module_partial: LightningModule = hydra.utils.instantiate(cfg.pa_callback.pa_module)
    # pa_module = pa_module_partial(classifier=deepcopy(vanilla_model.model).eval())
    # trainer_pa = hydra.utils.instantiate(cfg.pa_callback.trainers.pa_module.cpu)
    # trainer_pa.fit(
    #     model=pa_module, 
    #     train_dataloaders=pa_traindl,
    #     val_dataloaders=pa_traindl
    # )

    # assert torch.equal(pa_metric.betas, torch.tensor(pa_module.betas, dtype=float))
    # assert torch.allclose(pa_metric.logPAs, torch.tensor(pa_module.logPAs, dtype=float))


    # print("\nCheck that the PA and beta values are meaningful in the CPU:")
    # print("beta: ", pa_metric.betas)
    # print("logPA: ", pa_metric.logPAs)

    # print("\nCPU test passed.\n")
    # exit()

    # # ______________________________________ LIGHTNING ______________________________________
    # print("1")
    # pa_callback_partial = hydra.utils.instantiate(cfg.pa_callback.pa_callback)
    # pa_callback = pa_callback_partial(
    #     dataset = pa_traindl.dataset,
    #     cuda_devices = cfg.pa_callback.trainers.vanilla.ddp.devices
    # )
    # print("2")
    # pa_metric_partial = hydra.utils.instantiate(cfg.pa_callback.pa_metric)
    # pa_metric = pa_metric_partial(
    #                 dataset = pa_traindl.dataset,
    #                 processing_strategy = "lightning",
    #                 cuda_devices = cfg.pa_callback.trainers.vanilla.ddp.devices
    # )
    # print("3")
    # # We train the model with the callback and the metric inside:
    # trainer_vanilla_partial = hydra.utils.instantiate(cfg.pa_callback.trainers.vanilla.ddp)
    # trainer_vanilla = trainer_vanilla_partial(
    #     callbacks=[pa_callback]
    # )
    # print("4")
    # vanilla_model_partial = hydra.utils.instantiate(cfg.pa_callback.vanilla_model)
    # vanilla_model = vanilla_model_partial(
    #     metric = pa_metric
    # )
    # print("5")
    # trainer_vanilla.fit(
    #     vanilla_model,
    #     # I initialize the datamodule again because it's in DDP now
    #     datamodule=hydra.utils.instantiate(cfg.pa_callback.datamodules.main)
    # )
    # print("6")

    # """
    # The overall results of the callback and the metric should be exactly the same.
    # """
    # # Check PA results from the last epoch it was computed. For other epochs it has been checked from within the LightnignModule
    # assert torch.allclose(vanilla_model.metric.betas, vanilla_model.trainer.callbacks[0].pa_metric.betas)
    # assert torch.allclose(vanilla_model.metric.logPAs, vanilla_model.trainer.callbacks[0].pa_metric.logPAs)
    # assert torch.allclose(vanilla_model.metric.afr_pred, vanilla_model.trainer.callbacks[0].pa_metric.afr_pred)
    # assert torch.allclose(vanilla_model.metric.afr_true, vanilla_model.trainer.callbacks[0].pa_metric.afr_true)
    # assert torch.allclose(vanilla_model.metric.accuracy, vanilla_model.trainer.callbacks[0].pa_metric.accuracy)

    # # Check that the values are different
    # assert not torch.equal(vanilla_model.metric.betas, vanilla_model.metric.betas[torch.randperm(len(vanilla_model.metric.betas))])

    # # We also check the final results of the whole PA optimization (i.e. the PA selected for each call)    
    # assert torch.allclose(torch.tensor(vanilla_model.metric.log_beta), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_beta))
    # assert torch.allclose(torch.tensor(vanilla_model.metric.log_logPA), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_logPA))
    # assert torch.allclose(torch.tensor(vanilla_model.metric.log_AFR_pred), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_AFR_pred))
    # assert torch.allclose(torch.tensor(vanilla_model.metric.log_AFR_pred), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_AFR_pred))
    # assert torch.allclose(torch.tensor(vanilla_model.metric.log_accuracy), torch.tensor(vanilla_model.trainer.callbacks[0].pa_metric.log_accuracy))
    
    # print("\nCheck that the PA and beta values are meaningful for the lightning strategy:")
    # print("beta: ", vanilla_model.metric.betas)
    # print("logPA: ", vanilla_model.metric.logPAs)
    # exit()
    
    # # ______________________________________ CUDA ______________________________________
    # """
    # Now we will call the metric from outside again, and the PA module. 
    # """
    # print("7")
    # if dist.is_initialized():
    #     dist.destroy_process_group()
    
    # # Results should be the same (comparison within the model)
    # vanilla_model_partial = hydra.utils.instantiate(cfg.pa_callback.vanilla_model)
    # vanilla_model = vanilla_model_partial(metric=None)
    # model_to_eval = deepcopy(vanilla_model.model).eval()
    # model_to_eval2 = deepcopy(vanilla_model.model).eval()

    # # Initialize from here just in case
    # pa_metric_cuda = PosteriorAgreement(
    #                 dataset = pa_traindl.dataset,
    #                 beta0 = cfg.pa_callback.pa_module.beta0,
    #                 pa_epochs = cfg.pa_callback.trainers.pa_module.ddp.max_epochs,
    #                 processing_strategy = "cuda",
    #                 cuda_devices = cfg.pa_callback.trainers.vanilla.ddp.devices
    # )
    # pa_metric_cuda.update(model_to_eval, destroy_process_group=True)
    # print("8")

    # # Now compare with the PA module implementation
    # pa_module_partial: LightningModule = hydra.utils.instantiate(cfg.pa_callback.pa_module)
    # pa_module = pa_module_partial(classifier=model_to_eval2)
    # trainer_pa = hydra.utils.instantiate(
    #     cfg.pa_callback.trainers.pa_module.ddp
    # )
    # trainer_pa.fit(
    #     model=pa_module, 
    #     train_dataloaders=pa_traindl,
    #     val_dataloaders=pa_traindl
    # )
    # print("9")
    
    # print("\nCheck that the PA and beta values are meaningful in DDP:")
    # print("betas metric: ", pa_metric_cuda.betas)
    # print("betas module: ", torch.tensor(pa_module.betas, dtype=float))
    # print("logPA metric: ", pa_metric_cuda.logPAs[0, :])
    # print("logPA module: ", torch.tensor(pa_module.logPAs, dtype=float))

    # try:
    #     assert torch.equal(pa_metric_cuda.betas, torch.tensor(pa_module.betas, dtype=float)), "The betas are not the same between metric and module."
    #     assert torch.equal(pa_metric_cuda.logPAs[0, :], torch.tensor(pa_module.logPAs, dtype=float))
    #     assert pa_metric_cuda.afr_pred == pa_module.afr_pred.item(), "The AFR_pred is not the same between metric and module."
    #     assert pa_metric_cuda.afr_true == pa_module.afr_true.item(), "The AFR_true is not the same between metric and module."
    #     assert pa_metric_cuda.accuracy == pa_module.acc_pa.item(), "The accuracy is not the same between metric and module."
    # except:
    #     print("THEY ARE NOT EQUAL!!")

    # # exit()
    # print("\nTest passed.")

    


