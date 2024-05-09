from typing import List
from pytorch_lightning.callbacks import Callback
import torch

class ASS_Callback(Callback):
    """
    ASS accounts for average structural similarity.
    """

    def _SSIM(self, x_1: torch.Tensor, x_2: torch.Tensor):
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

    def _compute_ass(self, trainer, pl_module):
        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pa_metric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
        dataset = pa_metric_callback.pa_metric.dataset

        ass = torch.zeros(dataset.num_envs-1)
        for item in dataset:
            ssim = torch.tensor([
                self._SSIM(item[str(e)][0], item[str(e+1)][0])
                for e in range(dataset.num_envs-1)
            ])
            ass += ssim
        ass /= len(dataset)

        dict_to_log = {
            f"PA/ASS({e},{e+1})": ass[e].item()
            for e in range(dataset.num_envs-1)
        }
        self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=False)

    def on_train_start(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            self._compute_ass(trainer, pl_module)
    
    def on_test_start(self, trainer, pl_module):
        self._compute_ass(trainer, pl_module)