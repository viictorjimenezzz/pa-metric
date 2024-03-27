import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
from typing import Optional
import warnings
warnings.simplefilter("ignore")

# Tests to perform
from tests.test_pa.data_pipeline import *
from tests.test_pa.ddp import *
from tests.test_pa.pa_module import *
from tests.test_pa.pa_metric import *
from tests.test_pa.pa_callback import *

@hydra.main(
    version_base="1.3", config_path="../configs", config_name="test_pa.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Tests of the data pipeline.
    """
    # test_sampler(cfg)
    # test_dataloaders(cfg)
            
    """
    Tests of the parallelization strategy.
    """
    # test_ddp(cfg)

    """
    Tests of the PA module.
    """
    # test_pa_module(cfg)

    """
    Tests of the PA metric.
    """
    # test_basemetric(cfg)
    # test_pametric_cpu(cfg)
    # test_pametric_ddp(cfg)
    # test_pametric_logits(cfg)
    # test_accuracymetric(cfg)

    print("-----------------------------------------------")
    """
    Tests of the PA callback.
    """
    test_pa_callback(cfg)
        


if __name__ == "__main__":
    main()