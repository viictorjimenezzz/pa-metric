
from typing import Union, Optional
import os.path as osp
import warnings

import torch
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from pametric.datautils import MultienvDataset, LogitsDataset

def PosteriorAgreementDatasetPairing(
        dataset: MultienvDataset,
        strategy: Optional[str] = "label",
        pairing_csv: Optional[str] = None
    ):

    if not isinstance(dataset, MultienvDataset):
        warnings.warn("The dataset must be a MultienvDataset to work with the PA metric.")

    # If a custom pairing is provided, we use it.
    if pairing_csv:
        if osp.exists(pairing_csv):
            df = pd.read_csv(pairing_csv, dtype=int)
            dataset.permutation = [torch.tensor(df[col].values) for col in df.columns]
            return dataset
        else:
            warnings.warn("The pairing file does not exist. The pairing will be performed using the strategy and stored in such path.")
   
    # If the dataset has already been paired, then we don't need to do anything.
    print("Pairing observations...")
    for e in range(dataset.num_envs):
        if not torch.equal(torch.tensor(dataset.permutation[e], dtype=torch.long), torch.arange(len(dataset.dset_list[e]))):
            print("The dataset has already been permuted, so the pairing won't be performed. Try initializing the dataset again.")
            return dataset

    # If there are less than 2 envs, there is nothing to pair
    if dataset.num_envs < 2:
        return dataset
    
    if strategy == "nn":
        raise NotImplementedError
        # ref images is 0.
        # ref_images_flattened = reference_images.view(reference_images.shape[0], -1)
        # larger_images_flattened = larger_images.view(larger_images.shape[0], -1)

        # # Step 2: Compute the pairwise Euclidean distances
        # dists = torch.cdist(ref_images_flattened, larger_images_flattened, p=2)

        # # Step 3: Identify the nearest neighbors
        # _, indices = torch.min(dists, dim=1)

    
    else: # strategy == "label":
        labels0, labels1 = [
            torch.tensor([label for _, label in dataset.dset_list[i]]).long()
            for i in range(2)
        ]
        perm0, perm1 = _pair_optimize(labels0, labels1)

        new_permutations, new_permutations_filtered = [perm0, perm1], [perm0, perm1]

        # Add additional environments adjusted to the first two.
        if dataset.num_envs > 2:
            labels = labels0[perm0]
            add_labels_list = [torch.tensor([label for _, label in dataset.dset_list[i]]) for i in range(2, dataset.num_envs)]

            new_permutations = [None]*(dataset.num_envs-2)
            for i in range(len(add_labels_list)):
                new_perm = _pair_validate(labels, add_labels_list[i])
                new_permutations[i] = new_perm if new_perm is not None else None
            new_permutations = [perm0, perm1] + new_permutations
            new_permutations_filtered = [perm for perm in new_permutations if perm is not None]

        final_dataset = MultienvDataset(
            dset_list = [ds for i, ds in enumerate(dataset.dset_list) if new_permutations[i] is not None],
        )
        assert len(final_dataset.dset_list) == len(new_permutations_filtered), "The number of environments is not the expected."
        
        final_dataset.num_envs = len(new_permutations_filtered)
        final_dataset.permutation = new_permutations_filtered

    # It means that we want to save it in that location
    if pairing_csv:
        df = pd.DataFrame({f"env_{i}": new_permutations_filtered[i].tolist() for i in range(final_dataset.num_envs)})
        df.to_csv(pairing_csv, index=False)

    return final_dataset

def _pair_optimize(labels0: torch.Tensor, labels1: torch.Tensor):

    """
    Generates permutations for the first pair of environments so that their labels are correspondent.
    """
    
    labels_list = [labels0, labels1]
    inds = [torch.arange(len(labels0)), torch.arange(len(labels1))]

    # IMPORTANT: If the data is already paired, it could mean that not only the labels are paired but also the samples.
    # In such case, we don't want to touch it.
    if torch.equal(labels_list[0], labels_list[1]):
        return inds

    unique_labs = [labels.unique() for labels in labels_list] 
    common_labs = unique_labs[0][torch.isin(unique_labs[0], unique_labs[1])] # labels that are common to both environments

    final_inds = [[], []]
    for lab in list(common_labs):
        inds_mask = [inds[i][labels_list[i].eq(lab)] for i in range(2)] # indexes for every label
        if len(inds_mask[0]) >= len(inds_mask[1]):
            final_inds[0].append(inds_mask[0][:len(inds_mask[1])])
            final_inds[1].append(inds_mask[1])
        else:
            final_inds[0].append(inds_mask[0])
            final_inds[1].append(inds_mask[1][:len(inds_mask[0])])

    if final_inds[0] == [] or final_inds[1] == []:
        return [torch.empty(0), torch.empty(0)]
    else:
        return [torch.cat(final_inds[i]) for i in range(2)]

def _pair_validate(labels: torch.Tensor, labels_add: torch.Tensor):
    """
    Generates permutations for additional validation environments so that their labels are correspondent to the PA pair.
    - If the number of observations for certain labels is not enough, the samples are repeated.
    - If there are not observations associated with specific reference labels, the environment will be discarded.
    """
    if torch.equal(labels, labels_add):
        return torch.arange(len(labels))
    
    unique, counts = labels.unique(return_counts=True)
    sorted_values, sorted_indices = torch.sort(labels)
    unique_add, counts_add = labels_add.unique(return_counts=True)
    sorted_values_add, sorted_indices_add = torch.sort(labels_add)

    permuted = []
    for i in range(len(unique)):
        pos_add = (unique_add==unique[i].item()).nonzero(as_tuple=True)[0]
        if len(pos_add) == 0: # it means that that the label is not present in the second tensor
            warnings.warn("The label " + str(unique[i].item()) + " is not present in the tensor. Pairig is impossible, so the environment will not be used.")
            return None
        else:
            num = counts[i] # elements in the reference
            num_add = counts_add[pos_add.item()] # elements in the second tensor 
            diff = num_add - num
            vals_add = sorted_indices_add[counts_add[:pos_add].sum(): counts_add[:pos_add+1].sum()] # indexes of the second tensor
            if diff >= 0: # if there are enough in the second tensor, we sample without replacement
                permuted.append(vals_add[torch.randperm(num_add)[:num]])
            else: # if there are not enough, we sample with replacement (some samples will be repeated)
                permuted.append(vals_add[torch.randint(0, num_add, (num,))])

    perm = torch.cat(permuted)
    return perm[torch.argsort(sorted_indices)] # b => sorted_b' = sorted_a <= a


# import pyrootutils
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from src.data.components.diagvib_dataset import DiagVib6DatasetPA

# ds1 = DiagVib6DatasetPA(
#     mnist_preprocessed_path = "data/dg/mnist_processed.npz",
#     cache_filepath = "data/dg/dg_datasets/test_data_pipeline/train_singlevar0.pkl",
# )
# print("LENGTH OF THE DATASET 1: ", len(ds1))

# ds2 = DiagVib6DatasetPA(
#     mnist_preprocessed_path = "data/dg/mnist_processed.npz",
#     cache_filepath = "data/dg/dg_datasets/test_data_pipeline/train_singlevar1.pkl",
# )
# print("LENGTH OF THE DATASET 2: ", len(ds2))

# ds = MultienvDataset(dset_list=[ds1, ds2])

# paired_ds = PosteriorAgreementDatasetPairing(
#         dataset = ds,
#         strategy = "label",
#         pairing_csv = "./pairing_proves2.csv"
# )

# import ipdb; ipdb.set_trace()