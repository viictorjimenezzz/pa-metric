<div align="center">

# Posterior Agreement Metric

Metric computing the maximum of the Posterior Agreement kernel for classification tasks.


[![python](https://img.shields.io/badge/-Python3.9.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pytorch](https://img.shields.io/badge/Torchmetrics_0.11.0-792ee5?logo=lightning&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</div>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Implementation](#implementation)
- [How to Use](#instructions)
- [Examples](#examples)

## Overview

[This information will be expanded after publication]
    
###  Formal definition

Let $\boldsymbol{x}^{\prime}$ and $\boldsymbol{x}^{\prime \prime}$ be $N$-sized realizations of $\boldsymbol{X}$. With no prior information about the distribution over $\mathcal{C}$, the posterior agreement kernel for supervised $|\mathcal{C}|$-class classification tasks has the following expression


$$
k\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}^{\prime \prime}; \beta\right)=\log \left(\sum_{c \in \mathcal{C}} \frac{p\left(c \mid \boldsymbol{x}^{\prime}\right) p\left(c \mid \boldsymbol{x}^{\prime \prime}\right)}{p(c)}\right)
$$

Where $p\left(c \mid X\right)$ is the posterior probability distribution over the classes $\mathcal{C}$ given data $\boldsymbol{x}$. Its expression corresponds to a Gibbs distribution:

$$
    p(c \mid X)=\frac{\exp (\beta R(c, \boldsymbol{x}))}{\sum_{c^\prime \in \mathcal{C}} \exp \left(\beta R(c^\prime, \boldsymbol{x})\right)}
$$

where $\beta$ is the inverse temperature parameter and $R(c, \boldsymbol{x})$ the value of the risk for class $c$ given data $\boldsymbol{x}$.



The metric is obtained by maximizing the empirical PA kernel over $\beta$:

$$
\begin{aligned}
\text{PA}\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}^{\prime \prime}\right)=\underset{\beta}{\text{maximize}} & \frac{1}{N} k\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}^{\prime \prime}\right) . \\
\text { subject to } & \beta \geq 0
\end{aligned}
$$

### Key features

This implementation integrates seamlessly with the `torchmetrics` framework, offering:

- Compatibility with fully-supervised, semi-supervised, and unsupervised settings.
- Compatibility with a wide range of model selection and evaluation settings, including multi-environment validation/testing and model cross-validation.
- Integrated data pairing strategies, including label matching and feature-based pairing (e.g. nearest-neighbor, CCA, ...).
- Multi-device computation support through distributed-data-processing (DDP).
- Memory-efficient dataset evaluation with caching mechanism.
- Comprehensive logging and optimization process monitoring.

## Installation

[This information will be expanded after publication]

```bash
git clone [...]
pip install -r requirements.txt
```

## Implementation

As a `torchmetrics.Metric` subclass, the implementation provides three key methods:

1. `__init__()`: Handles dataset pairing and multi-processing configuration.
2. `update()`: Manages classifier initialization, CPU result preallocation, beta optimization, and classifier cleanup.
3. `compute()`: Selects the epoch maximizing the kernel and retrieves associated metrics.

The PA metric requires the model's probabilistic output (logits) as input. You can provide this data in two ways:

1. Direct logits input using `pametric.datautils.LogitsDataset`, which accepts multiple logits tensors representing classifier evaluations across different environments.

2. Dataset and classifier input using `pametric.datautils.MultienvDataset` along with a `torch.nn.Module` classifier. This approach performs model evaluation within the metric itself. The metric requires either:
   - One classifier with two environments for computing $\text{PA}\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}^{\prime \prime}\right)$, or
   - Two classifiers with one environment for computing $\text{PA}\left(\gamma^{\prime}, \gamma^{\prime \prime}\right)$.

Additional classifiers and/or environments will be used for validation, and the kernel and performance metrics values will be also provided for these.

## How to Use

Two versions of the metric have been implemented:

| Feature | `PosteriorAgreementBase` | `PosteriorAgreement` |
|---------|------------------------|-------------------|
| Environment Support | Two environments only | Multiple environments and classifiers |
| Processing Capability | Single process | Multi-processing supported |
| Optimizer Support | Limited (SGD, Adam) | Custom optimizer |
| Compatible with `pytorch-lightning` | No | Yes |

### Simple version

`PosteriorAgreementBase` implementation accepts the following parameters:

- `dataset`: A `LogitsDataset` or `MultienvDataset` instance with two environments.
- `beta0`: The initial value for the optimization parameter `beta`.
- `optimizer_name`: Specifies the optimization algorithm, accepting either `"SGD"` or `"Adam"`.
- `optimizer_lr`: Defines the learning rate for the optimizer, with a default value of `0.1`.
- `pairing_strategy`: Determines how observations from different environments are paired together. The available strategies are:
    - `"label"`: The default strategy. This option assumes the dataset is already paired through a controlled data generation process, requiring no additional configuration.
    - `"nn"`: Implements nearest-neighbor pairing in the feature space using [FAISS](https://github.com/facebookresearch/faiss). This strategy requires a `feature_extractor` parameter of type `torch.nn.Module` to generate latent representations.
    - `"cca"`: Implements canonical correlation analysis for pairing observations, which also needs a `feature_extractor`.
- `pairing_csv`: Provides a path to store and load pairing results. This parameter optimizes performance by allowing the reuse of computationally expensive pairing calculations across multiple runs.

| Parameter | Description | Default |
|-----------|-------------|---------------|
| `dataset` | A `LogitsDataset` or `MultienvDataset` instance with two environments. | Required |
| `pa_epochs` | Number of `beta` optimization epochs. | Required |
| `beta0` | The initial value for the optimization parameter `beta`. | `1.0` |
| `optimizer_name` | Specifies the optimization algorithm, accepting either `"SGD"` or `"Adam"`. | `"Adam"` |
| `optimizer_lr` | Defines the learning rate for the optimizer. | `0.1` |
| `pairing_strategy` | Determines how observations from different environments are paired together. The available strategies are:<br>- `"label"`: This option assumes the dataset is already paired through a controlled data generation process, requiring no additional configuration.<br>- `"nn"`: Implements nearest-neighbor pairing in the feature space using [FAISS](https://github.com/facebookresearch/faiss). This strategy requires a `feature_extractor` parameter of type `torch.nn.Module` to generate latent representations.<br>- `"cca"`: Implements canonical correlation analysis for pairing observations, which also needs a `feature_extractor`. | `"label"` |
| `pairing_csv` | Provides a path to store and load pairing results. This parameter optimizes performance by allowing the reuse of computationally expensive pairing calculations across multiple runs. | None |

### Full version

`PosteriorAgreement` extends the simple version with additional functionality through these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------------|
| `optimizer` | Accepts a custom optimizer for the beta parameter. The optimizer must be provided as a partially-instantiated `torch.optim.Optimizer` object (e.g. with `functools.partial`), with parameter initialization deferred. | `torch.optim.Adam` with `lr = 0.1` |
| `batch_size` | Controls the batch size for the data evaluation dataloader. | `16` |
| `num_workers` | Specifies the number of worker processes for the data evaluation dataloader. The system will automatically configure this value if not explicitly set. | Auto-configured |
| `processing_strategy` | Determines the computation approach with three options:<br>- `"cpu"`: Executes the metric computation in a single local process.<br>- `"cuda"`: Enables parallel computation across multiple CUDA devices, supporting both ongoing optimization and new process groups. Custom configuration is possible by overriding `_multiprocessing_conf()`.<br>- `"lightning"`: Enables integration with the PyTorch Lightning framework, where device management is handled by the Lightning `Trainer` through `local_rank`. | `"cpu"` |
| `cuda_devices` | Specifies the CUDA devices for parallel processing. This parameter is mandatory when using either the `"lightning"` or `"cuda"` processing strategies. | Required for `"cuda"` and `"lightning"` strategies |

The full version of the metric can be integrated within a `pytorch-lightning` setup by setting `processing_strategy="lightning"`. Additionally, PA can also be obtained without modifying the structure of the `LightningModule`, but simply adding the `pametric.lightning.callbacks.metric.PA_Callback`. For additional integration options and utilities, refer to the `pametric/lightning/` directory.

## Examples

### Simple version

```python
from omegaconf import OmegaConf
from pametric import PosteriorAgreementBase, LogitsDataset

# Load arguments and initialize:
kwargs = OmegaConf.to_container(OmegaConf.load("basemetric.yaml"), resolve=True)
pa_metric = PosteriorAgreementBase(
    dataset = LogitsDataset([logits0, logits1], y)
    **kwargs
)

# Compute the metric:
results = pa_metric(classifier=model)
log_PA, beta = results.get("logPA"), results.get("beta")
```

### Full version

Similar setup as the simple version, but more configuration possibilities:

```python
import functools
from omegaconf import OmegaConf
from pametric import PosteriorAgreement, MultienvDataset

# Load arguments and initialize:
kwargs = OmegaConf.to_container(OmegaConf.load("metric.yaml"), resolve=True)
# optimizer = functools.partial(OptimizerClass, **kwargs.pop("optimizer")) also possible
pa_metric = PosteriorAgreement(
    dataset = MultienvDataset([ds0, ds1, ...]) # additional ds2, ds3... for validation
    **kwargs
)

# Compute the metric:
results = pa_metric(classifier=model)
log_PA, beta = results.get("logPA"), results.get("beta")
```

For instance, it is possible to use two classifiers, which will be evaluated on the first environment, to optimize `beta` within a cross-validation setting:

```python
pa_metric = PosteriorAgreement(
    dataset = MultienvDataset([ds0, ...]) # additional ds1, ds2... for validation
    **kwargs
)

results = pa_metric(model0, model1)
log_PA, beta = results.get("logPA"), results.get("beta")
```

In a `pytorch-lightning` workflow, you must add the metric call at the end of the optimization epoch:

```python
# At initialization
kwargs = {**kwargs, processing_strategy="lightning"} # make sure
self.pa_metric = PosteriorAgreement(
    dataset = MultienvDataset(...)
    **kwargs
)

# At `on_train_epoch_end()` and/or `on_test_epoch_end()`:
results = self.pa_metric(
    classifier=pl_module.model,
    local_rank=trainer.local_rank
)
```

Alternatively, you can simply rely on the callback to manage data, models, and multi-processing configuration automatically:

```python
from omegaconf import OmegaConf
from pametric.lightning.callbacks import PA_Callback

# Initializing the callback:
kwargs = OmegaConf.to_container(OmegaConf.load("pa_callback.yaml"), resolve=True)
pa_callback = PA_Callback(**kwargs)

# Adding it to the trainer
trainer = Trainer(..., callbacks=[..., pa_callback])
```
