import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


#--------------------------------------------------
#print(training_data.targets)
from abc import ABCMeta, abstractmethod
from sklearn.model_selection._split import BaseCrossValidator


from typing import Optional, List, Literal
import csv

class Custom_CV(BaseCrossValidator):
    """Custom CV implementation using pytorch and designed to substitute sklearn method.
    
    Args:
        train_ds (Dataset): Dataset with the whole training data.
        config_csv (str): Path to csv file with the splitting configuration.
        test_ds (Optional[Dataset]): Dataset with the whole test data. If provided, all splits will be use for training.
        shuffle (Optional[Literal["random", "sequential", "paired"]]): Shuffle strategy for the test split, either "random", "sequential" (i.e. not shuffled) and "paired", by which samples of different splits are label-correspondent.
        random_state (Optional[int], optional): Random state. Defaults to 123.

    Returns: 
        fold_counter: Informs of the fold index. Every fold can contain multiple training groups.
        group_info: A tuple indicating (group index, total number of groups).
        train_ind_tensor: A tensor with the indexes for the training dataset, that will constitute the train split.
        test_ind_list: A list containing two tensors of indexes for the test dataset (if provided) or the train dataset, that will constitute the two environments of the test split. If finally only test dataset wants to be used, use ConcatDataset.

    Example of use: No test dataset is provided, and the same split is used for testing each fold.
    ds0,ds1,ds2,ds3,ds4
    0,1,1,2,2
    0,1,2,1,2
    0,2,1,1,2
    0,1,2,2,1
    0,2,1,2,1
    0,2,2,1,1

    cv = Custom_CV(train_ds=BigDS, config_csv=..., shuffle="paired")
    for i, (fold, group_info, train_idx, test_idx) in enumerate(cv.split()):
        print(f"\nFold {fold}")
        group_idx, num_groups = group_info # useful to average the metrics for example.

        train_ds = Subset(BigDS, train_idx)
        test_ds1 = Subset(BigDS, test_idx[0])
        test_ds2 = Subset(BigDS, test_idx[1])

        # train and evaluate (metric)

    It is important to notice that the training data is always the same size, but the size of the test data varies slightly in the "paired" setup due to the non-uniformity of the target distribution. 
    The test indexes for every element of the group are the same, only the training varies.

    """

    def __init__(self,
                 train_ds: Dataset,
                 config_csv: str,
                 test_ds: Optional[Dataset] = None,
                 shuffle: Optional[Literal["random", "sequential", "paired"]] = "random",
                 random_state: Optional[int] = 123):
        
        self.train_dataset = train_ds
        if test_ds:
            self.test_dataset = test_ds
        else:
            self.test_dataset = None

        # Turn CSV into a list of lists
        self.configlist = []
        with open(config_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                self.configlist.append([int(item) for item in row])

        self.n_folds = self.get_n_folds()
        self.n_splits = self.get_n_splits()

        if shuffle not in ["random", "sequential", "paired"]:
            raise ValueError("shuffle must be 'random', 'sequential' or 'paired'; got {0}".format(shuffle))

        if shuffle != "sequential" and type(random_state) is not int:
            raise ValueError("random_state must be an integer when shuffle is not 'sequential'; got {0}".format(random_state)) 
        
        self.shuffle = shuffle

    def get_n_folds(self):
        """Returns the number of folds in the cross-validator."""
        if len(self.configlist) < 1:
            raise ValueError("Configuration list is empty.")
        
        return len(self.configlist)
    
    def get_n_splits(self):
        """Returns the number of splits in the cross-validator."""
        n_splits_0 = len(self.configlist[0])
        for f in range(self.n_folds): # loop to check before training
            n_splits = len(self.configlist[f])
            if n_splits != n_splits_0:
                raise ValueError("Configuration must specify the same number of splits for each fold.")
            if n_splits < 1:
                raise ValueError("Configuration must include at least one split for each fold.")
            
        return n_splits_0
    
    def _next_split_config(self, bool_test_ds: bool = False):
        """Generates the configuration dictionary for the next split."""

        for f in range(self.n_folds): # loop to check before training
            if not bool_test_ds and self.n_splits < 2:
                raise ValueError("Since no test set is provided, configuration must include at least two splits for each fold.")
            
        for f in range(self.n_folds):
            # Get to same numbers for consistency
            if not bool_test_ds: # If we dont specify a test dataset, take smaller group index as test
                ind_zero = 0
            else:
                ind_zero = 1
            group_inds = sorted(list(set(self.configlist[f])))
            map_to_inds = {el: i + ind_zero for i, el in enumerate(group_inds)}
            group_inds = [map_to_inds[el] for el in self.configlist[f]]

            # Get group dictionary
            group_dict = {}
            for i, el in enumerate(group_inds):
                if el not in group_dict:
                    group_dict[el] = []
                group_dict[el].append(i)
            if bool_test_ds:
                group_dict[0] = None

            yield group_dict

    def get_n_samples(self, dataset: Dataset):
        """Returns the number of samples in the dataset."""
        return len(dataset)

    def paired_indexes(self, dataset:Dataset, pair_splits: int):
        """Computes a list of indexes for every split in a way that targets match."""

        n_samples = self.get_n_samples(dataset)
        inds = torch.arange(n_samples)

        try:
            labs = dataset.targets
        except AttributeError: # when its not a full Dataset, but a Subset or ConcatDataset
            labs = torch.tensor([dataset.__getitem__(i)[1] for i in range(len(dataset))])

        unique_labs = labs.unique()
        inds_mask = [inds[labs.eq(unique_lab.item())] for unique_lab in unique_labs] # indexes for every label
        inds_mask = [mask[torch.randperm(mask.size(0))] for mask in inds_mask] # randomly permute the indexes for every label (unnecessary)
        n_split_lab = min([mask.size(0) // pair_splits for mask in inds_mask])
        split_permutation = torch.randperm(n_split_lab*len(unique_labs))

        indexes = [
            torch.cat([mask[n*n_split_lab:(n+1)*n_split_lab] for mask in inds_mask]
                        )[split_permutation] # same permutation to all the splits (to mix labels but keep correspondence)
            for n in range(pair_splits)]

        return indexes

        
    def split(self):
        """Generate indices to split data.""" 

        # Get train indexes list. These are only training if no test_dataset is provided.
        n_train_samples = self.get_n_samples(self.train_dataset)
        n_samples_split = n_train_samples // self.n_splits
        train_inds = torch.arange(n_train_samples)
        if self.shuffle != "sequential": # if sequential, both training and test are produced in a sequential way
            train_inds = train_inds[torch.randperm(n_train_samples)]
        train_indexes = [train_inds[n*n_samples_split:(n+1)*n_samples_split] for n in range(self.n_splits)]

        # Get test indexes list. These are only generated if there is a test_dataset.
        if self.test_dataset: # the specified splits only concern the training dataset
            n_test_samples = self.get_n_samples(self.test_dataset)
            n_samples_test_split = n_test_samples // 2 # because we want the test to be divived in two
            test_inds = torch.arange(n_test_samples)
            
            if self.shuffle == "paired": # we pair the test indexes, train indexes are randomly permuted
                test_ind_list = self.paired_indexes(self.test_dataset, pair_splits = 2)

            else: # either random or sequential
                if self.shuffle == "random": # test is random, train is random
                    test_inds = test_inds[torch.randperm(n_test_samples)]
                test_ind_list = [test_inds[n*n_samples_test_split:(n+1)*n_samples_test_split] for n in range(2)]

        fold_counter = -1
        for sconfig in self._next_split_config(True if self.test_dataset else False):
            fold_counter += 1
            group_names = sorted(list(sconfig.keys()))[1:] # all but 0, which is for the test split
            num_groups = len(group_names)

            if not self.test_dataset: # the test_ind_list with 2 elements must be generated
                test_inds = torch.cat([train_indexes[i] for i in sconfig[0]]) # join train splits specified for testing
                if self.shuffle == "paired":
                    test_ind_list = self.paired_indexes(Subset(self.train_dataset, test_inds), pair_splits = 2) # indexes wrt Subset
                    test_ind_list = [test_inds[test_ind_list[i]] for i in range(2)] # indexes wrt self.train_dataset
                else:
                    if self.shuffle == "random":
                        test_inds = test_inds[torch.randperm(len(test_inds))]

                    n_samples_test_split = len(test_inds) // 2 # impose two groups
                    test_ind_list = [test_inds[n*n_samples_test_split:(n+1)*n_samples_test_split] for n in range(2)]

            for group in group_names:
                train = torch.cat([train_indexes[i] for i in sconfig[group]])
                yield fold_counter, (group-1, num_groups), train, test_ind_list