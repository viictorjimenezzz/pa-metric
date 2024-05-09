from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import torch

class MeasureDataset_Callback(Callback):
    """
    Base callback to compute metrics associated with the initial dataset. 
    These metrics are logged one time only, at the beggining of train and test.
    """
    metric_name: str = "metric_name"

    def _metric(self, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
        """
        To override with the computation of the metric.
        """
        pass

    def _compute_average(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pa_metric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
        dataset = pa_metric_callback.pa_metric.dataset
        self.num_envs = dataset.num_envs

        sum_val = torch.zeros(dataset.num_envs-1)
        for item in dataset:
            sum_val += torch.tensor([
                self._metric(item['0'][0], item[str(e)][0])
                for e in range(1, dataset.num_envs)
            ])
        import ipdb; ipdb.set_trace()
        return sum_val / len(dataset)
    
    def _log_average(self, average_val: torch.Tensor) -> None:
        dict_to_log = {
            f"PA(0,{e})/{self.metric_name}": average_val[e].item()
            for e in range(1, self.num_envs)
        }
        self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=False)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.current_epoch == 0:
            average_val = self._compute_average(trainer, pl_module)
            self._log_average(average_val)
    
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        average_val = self._compute_average(trainer, pl_module)
        self._log_average(average_val)

        
class ASS_Callback(MeasureDataset_Callback):
    """
    ASS accounts for average structural similarity.
    """

    metric_name = "ASS"

    def _metric(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """
        Computes structural similarity index (SSIM) for the pair of images (x, y):
            SSIM(x, y) = (2*ms_x*ms_y + cs_1)*(2*ds_xy + cs_2)/[(ms_x**2 + ms_y**2 + cs_1)*(ds_x**2 + ds_y**2 + cs_2)]

        where ms_x, ms_y and ds_x, ds_y, ds_xy are the means and covariances, respectively, and c_1, c_2 are constants
        that stabilize the division.
        """
        
        cs = [(0.01 * 255) ** 2, (0.03 * 255) ** 2] # arbitrary
        ms = [x_1.mean(), x_2.mean()]
        ds = [x_1.var(unbiased=False), x_2.var(unbiased=False), ((x_1 - ms[0]) * (x_2 - ms[1])).mean()]
        num = (2*ms[0]*ms[1] + cs[0]) * (2*ds[2] + cs[1])
        den = (ms[0]**2 + ms[1]**2 + cs[0]) * (ds[0] + ds[1] + cs[1])
        return num.item() / den.item()