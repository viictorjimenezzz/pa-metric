from typing import Optional
import os.path as osp
import warnings
import time

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from faiss import IndexFlatL2, IndexIVFFlat, METRIC_L2
from pametric.datautils import MultienvDataset

def PosteriorAgreementDatasetPairing(
        dataset: MultienvDataset,
        strategy: Optional[str] = "label",
        pairing_csv: Optional[str] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
    ):

    # Available strategies at the moment:
    assert strategy in ["label", "label_nn", "nn_IVFFlat", "nn_L2"], "The strategy must be either 'label', 'label_nn','nn_IVFFlat' or 'nn_L2'."

    # Check feature extractor:
    if strategy in ["nn_IVFFlat", "nn_L2"]:
        assert feature_extractor is not None, "A feature extractor must be provided for the NN strategy."

    # Dataset must be Multienv Dataset
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
    for e in range(dataset.num_envs):
        if not torch.equal(torch.tensor(dataset.permutation[e], dtype=torch.long), torch.arange(len(dataset.dset_list[e]))):
            print("The dataset has already been permuted, so the pairing won't be performed. Try initializing the dataset again.")
            return dataset

    # If there are less than 2 envs, there is nothing to pair
    if dataset.num_envs < 2:
        return dataset

    print("\nPairing observations...")

    if strategy[:2] == "nn":
        # We extract the fectures vector using the desired model:
        start_time = time.time()
        print("\nFeature extraction started...")
        features_list = [
            _FeatureExtractor(ds, feature_extractor)
            for ds in dataset.dset_list        
        ]
        print(f"Time spent extracting data features: ~ {(time.time() - start_time) // 60} min")

        permutations = [torch.arange(len(dataset.dset_list[0]))] + [
            NNFaiss(features_list[0], features_list[i], strategy.split("_")[-1])
            for i in range(1, dataset.num_envs)
        ]

        # Generate final dataset
        final_dataset = MultienvDataset([ds for ds in dataset.dset_list])
        final_dataset.num_envs = dataset.num_envs
        final_dataset.permutation = permutations
        
        # It means that we want to save it in that location
        if pairing_csv:
            df = pd.DataFrame({f"env_{i}": permutations[i].tolist() for i in range(dataset.num_envs)})
            df.to_csv(pairing_csv, index=False)

    elif strategy == "label_nn":
        # We extract the fectures vector using the desired model:
        start_time = time.time()
        print("\nFeature extraction started...")
        features_list = [
            _FeatureExtractor(ds, feature_extractor)
            for ds in dataset.dset_list        
        ]
        print(f"Time spent extracting data features: ~ {(time.time() - start_time) // 60} min")

        # Extract labels and inds of the first two environments
        labels_list = [
            torch.tensor([label for _, label in ds]).long()
            for ds in dataset.dset_list
        ]
        inds = [
            torch.arange(len(labs))
            for labs in labels_list
        ]
        unique_labs = [labels.unique() for labels in labels_list] 
        common_labs = unique_labs[0][torch.isin(unique_labs[0], unique_labs[1])] # labels that are common in the first two environments
        
        permutations = [torch.arange(len(dataset.dset_list[0]))]
        for env in range(1, dataset.num_envs):
            perm_env = torch.zeros(len(dataset.dset_list[env]), dtype=torch.long)
            for lab in list(common_labs):
                inds_mask = [inds[i][labels_list[i].eq(lab)] for i in [0, env]] # indexes for every label

                perm_env_lab = NNFaiss(
                    features_list[0][inds_mask[0]],
                    features_list[env][inds_mask[1]],
                    "L2"
                )
                perm_env[inds_mask[0]] = inds_mask[1][perm_env_lab]


            permutations += [perm_env]

        # Generate final dataset
        final_dataset = MultienvDataset([ds for ds in dataset.dset_list])
        final_dataset.num_envs = dataset.num_envs
        final_dataset.permutation = permutations
        
        # It means that we want to save it in that location
        if pairing_csv:
            print("\nTHERE MIGHT BE A PROBLEM WITH THE LENGTH? ", [len(perm) for perm in permutations])
            df = pd.DataFrame({f"env_{i}": permutations[i].tolist() for i in range(dataset.num_envs)})
            df.to_csv(pairing_csv, index=False)

    else: #Simply label pairing, strategy == "label":
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


def _FeatureExtractor(
        dataset: Dataset,
        feature_extractor: torch.nn.Module
    ):
    """
    We obtain a feature vector from an image using the very same model that is used to
    train the architecture, but pretrained with the SOTA weights.
    """
    feature_extractor.eval()
    feature_extractor

    features = []
    with torch.no_grad():
        for batch_x, _ in DataLoader(dataset, batch_size=16, shuffle=False):
            feature_vec = feature_extractor(batch_x)
            features.append(feature_vec.cpu().numpy().squeeze().astype('float32'))
    return np.concatenate(features)

def NNFaiss(
        features_ref:  np.array,
        features_large: np.array,
        index: str = "IVFFlat"
    ):
    # See https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    len_large_ds, len_ref_ds = features_large.shape[0], features_ref.shape[0]
    dim_vecs = features_large.shape[1]

    # TODO: IndexIVFFlat
    # Memory limit of 1 GB for RAM. Each float32 takes 4 bytes.
    RAM_limit = 1 * (1024 ** 3) // (dim_vecs*4) # TODO: adjust

    """
    From the guidelines we deduce that:
    len_large_ds >= n_train >= train_factor*n_clusters = train_factor*cluster_factor*sqrt(len_large_ds)

    The values for the multiplicative factors are:
        - train_factor = 40:256
        - cluster_factor = 4:16

    Then the minimum n_train is 40*4*sqrt(len_large_ds) <= len_large_ds <=> 160 <= sqrt(len_large_ds) <=> 25600 <= len_large_ds
    """

    if index == "IVFFlat" and len_large_ds >= 25600:
        """
        Then we train a IVFFlat index.
        """
        # Deciding the number of clusters and index training samples.
        n_train_samples, n_clusters_samples = [1], [2]
        for cluster_factor in range(4, 16):
            for train_factor in range(40, 256): # BUG fix: From 30 to ?? per warning suggestion
                n_clusters = int(cluster_factor*np.sqrt(len_large_ds)) 
                n_train = min(
                        RAM_limit, 
                        train_factor*n_clusters, # As per the guidelines
                        len_large_ds # Length of the dataset
                )
                if n_train == RAM_limit:
                    break
                elif n_train >= 40*n_clusters: # safeguard
                    n_train_samples.append(n_train)
                    n_clusters_samples.append(n_clusters)

        pos_max = np.argmax(n_clusters_samples) # the first one will have the fewest number of samples.
        n_train = n_train_samples[pos_max]
        n_clusters = n_clusters_samples[pos_max]
        
        index = IndexIVFFlat(
            IndexFlatL2(dim_vecs),
            dim_vecs,  # Dimension of the vectors
            n_clusters, #int(np.sqrt(large_ds.shape[0])), # Number of clusters
            METRIC_L2 # L2 distance
        )
        
        start_time = time.time()
        # print("\nIndex training started...")
        train_subset = features_large[np.random.choice(len_large_ds, n_train, replace=False), :]
        index.train(train_subset)
        # print(f"Time spent training: ~ {(time.time() - start_time) // 60} min")

    else:
        """
        Then we train a L2 Flat index.
        """
        index = IndexFlatL2(dim_vecs)

    # Now we can add the data to the index by batches: (10 MB limit)
    start_time = time.time()
    # print("\nIndex build started...")
    batch_size = min(10*(1024 ** 2) // (dim_vecs*4), len_large_ds)
    for i in range(0, len_large_ds, batch_size):
        index.add(
            features_large[i:min(i + batch_size, len_large_ds), :]
        )
    # print(f"Time spent building index: ~ {(time.time() - start_time) // 60} min")

    # Quality check:
    _, inds = index.search(features_ref[0, :].reshape(1, -1), k=1) 
    if inds[0].item() == -1:
        raise ValueError("\nThe index has not been trained properly. Try changing the centroid number and the number of training samples.")

    # Perform search for each vector of the reference ds.
    start_time = time.time()
    # print("\nIndex search started...")
    perm1 = torch.tensor(
        np.array([
        index.search(features_ref[i, :].reshape(1, -1), k=1)[1][0].item() # closest element, with repetition
        for i in range(len_ref_ds)
        ])
    )
    # print(f"Time spent searching: ~ {(time.time() - start_time) // 60} min")
    return perm1
