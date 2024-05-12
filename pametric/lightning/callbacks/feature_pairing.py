from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from pametric.datautils import MultiEnv_collate_fn

import numpy as np
from copy import deepcopy
from pametric.lightning.callbacks import SplitClassifier

from faiss import IndexFlatL2, IndexIVFFlat, METRIC_L2

class FeaturePairing_Callback(Callback):
    """
    Performs pairing between samples of different environments based on the
    feature extractor resulted after each epoch. The pairing is performed with FAISS
    implementation of NN (either exact or approximate).
    """

    def __init__(self, index: str = "IVFFlat"):
        super().__init__()
        self.index = index

    def _extract_features(
            self,
            dataset: Dataset,
            feature_extractor: torch.nn.Module
        ):
        len_ds = len(dataset)
        features = torch.zeros(( 
            self.num_envs,
            len_ds,
            feature_extractor.forward(dataset[0]['0'][0].unsqueeze(0), extract_features = True).size(1)
        ))

        dataloader = DataLoader(
                    dataset = dataset,
                    collate_fn = MultiEnv_collate_fn,
                    batch_size = self.batch_size,
                    num_workers = 0, 
                    pin_memory = False,
                    sampler = SequentialSampler(dataset),
                    drop_last=False,
        )
        with torch.no_grad():
            for bidx, batch in enumerate(dataloader):
                features[:, bidx*self.batch_size: min((bidx+1)*self.batch_size, len_ds), :] = torch.stack(
                    [
                    feature_extractor.forward(batch[e][0], extract_features = True).squeeze()
                    for e in list(batch.keys())
                ],
                )
            return features
        
    def NN_FAISS(
            self,
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
            
            # print("\nIndex training started...")
            train_subset = features_large[np.random.choice(len_large_ds, n_train, replace=False), :]
            index.train(train_subset)
            # print(f"Time spent training: ~ {(time.time() - start_time) // 60} min")

        else:
            """
            Then we use a L2 Flat index.
            """
            index = IndexFlatL2(dim_vecs)

        # Now we can add the data to the index by batches: (10 MB limit)
        # start_time = time.time()
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
        # start_time = time.time()
        # print("\nIndex search started...")
        perm1 = torch.tensor(
            np.array([
            index.search(features_ref[i, :].reshape(1, -1), k=1)[1][0].item() # closest element, with repetition
            for i in range(len_ref_ds)
            ])
        )
        # print(f"Time spent searching: ~ {(time.time() - start_time) // 60} min")
        return perm1


    def _pair(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        # Get the model and split it into feature extractor and classifier
        model_to_eval = SplitClassifier(
            net = deepcopy(pl_module.model.net),
            net_name = pl_module.model.net_name
        ).eval()

        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pa_metric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
        dataset = pa_metric_callback.pa_metric.dataset
        self.num_envs = dataset.num_envs
        self.batch_size = pa_metric_callback.pa_metric.batch_size

        # Extract the tensor of features
        features = self._extract_features(dataset, model_to_eval)

        permutation = [np.arange(len(dataset))]
        for e in range(features.size(0) - 1):
            permutation.append(
                self.NN_FAISS(
                    features_ref = features[e].cpu().numpy(),
                    features_large = features[e + 1].cpu().numpy(),
                    index = self.index
                )
            )
        
        # Assign the permutation
        pa_metric_callback.pa_metric.dataset.permutation = [torch.from_numpy(perm).to(dtype=torch.int) for perm in permutation]
        
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self._pair(trainer, pl_module)