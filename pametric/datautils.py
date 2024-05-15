import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from typing import List

# Map the PyTorch dtype to its size in bytes
dtype_size_mapping = {
    torch.float32: 4,  # or torch.float
    torch.float64: 8,  # or torch.double
    torch.float16: 2,  # or torch.half
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,    # or torch.short
    torch.int32: 4,    # or torch.int
    torch.int64: 8     # or torch.long
}

def adjust_batch_size(image_batch_size: int, sample_image: torch.Tensor, sample_logit: torch.Tensor):
    """
    Returns the batch size for a logits Dataloader, adjusted by the memory ratio between images and logits.
    """
    # Get memory of tensor
    image_channels, image_height, image_width = sample_image.shape
    image_dtype_size = dtype_size_mapping.get(sample_image.dtype)
    memory_per_image = image_width * image_height * image_channels * image_dtype_size
    
    # Get memory of logits
    num_classes = sample_logit.shape[-1]
    logit_dtype_size = dtype_size_mapping.get(sample_logit.dtype)
    memory_per_logit_set = num_classes * logit_dtype_size

    # Batch size adjusted by image-to-logit memory ratio
    logits_to_images_ratio = memory_per_image / memory_per_logit_set
    return int(image_batch_size * logits_to_images_ratio)

class MultienvDataset(Dataset):
    """
    We assume datasets return a tuple (input tensor, label).
    """

    def __init__(self, dset_list: List[Dataset]):
        self.dset_list = dset_list
        self.num_envs = len(dset_list)
        self.permutation = [torch.arange(len(ds)).tolist() for ds in self.dset_list]

    def __len__(self):
        return min([len(perm) for perm in self.permutation])
 
    def __getitem__(self, idx: int):
        return {str(i): dset[self.permutation[i][idx]] for i, dset in enumerate(self.dset_list)}
    
    def __getitems__(self, indices: List[int]):
        """
        When I request several items, I prefer to get a tensor for each dataset.
        """
        # Is there a way to do it without multiplicating the calls to __getitem__?
        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = tuple([torch.stack([self.__getitem__(idx)[str(i)][0] for idx in indices]), 
                                    torch.tensor([self.__getitem__(idx)[str(i)][1] for idx in indices])])

        return output_list
    
    def __getlabels__(self, indices: List[int]):

        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = torch.tensor([self.__getitem__(idx)[str(i)][1] for idx in indices]) 

        return output_list
    
    def Subset(self, indices: List[int]):
        """
        Returns a new MultienvDataset object with the subset of the original dataset.
        """
        subset_items = self.__getitems__(indices)
        return MultienvDataset([TensorDataset(*env_subset) for env_subset in subset_items])


class LogitsDataset(Dataset):
    """
    TorchDataset wrapper for logits computation in the PA metric.
    """
    def __init__(self, logits: List[Tensor], y: Tensor) -> None:
        self.num_envs = len(logits)
        self._check_input(logits, y)
        self.logits = logits
        self.y = y

    def _check_input(self, logits: List[Tensor], y: Tensor) -> None:
        assert self.num_envs == len(logits), "Must add a logit for each environment"
        assert all(logits[0].size(0) == logit.size(0) for logit in logits), "Size mismatch between logits"
        assert all(y.size(0) == logit.size(0) for logit in logits), "Size mismatch between y and logits"

    def __len__(self):
        return self.logits[0].size(0)

    def __additem__(self, logits: List[Tensor], y: Tensor) -> None:
        """
        This method is slow, because it's concatenating tensors, so it should be avoided whenever possible.
        """
        self._check_input(logits, y)
        self.y = torch.cat([self.y, y])
        
        for i in range(self.num_envs):
            self.logits[i] = torch.cat([self.logits[i], logits[i]]) 

    def __getitem__(self, index: int):
        return {str(i): tuple([self.logits[i][index], self.y[index]]) for i in range(self.num_envs)}
    
    def __getitems__(self, indices: List[int]):
        """
        When I request several items, I prefer to get a tensor for each dataset.
        """
        return [tuple([self.logits[i][indices], self.y[indices]]) for i in range(self.num_envs)]
    

def MultiEnv_collate_fn(batch: List):
    """
    Collate function for multi-environment datasets and multi-environment models.

    The output is of the form:

    batch_dict = {
        "env1_name": [x1, y1],
        "env2_name": [x2, y2],
        ...
    """

    batch_dict = {}
    for env in batch[0]:
        batch_dict[env] = [
            torch.stack([b[env][0] for b in batch]),
            torch.tensor([b[env][1] for b in batch]) if not isinstance(batch[0][env][1], torch.Tensor) else torch.cat([b[env][1] for b in batch]),
        ]

    return batch_dict