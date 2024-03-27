

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

__all__ = ["plot_multienv", "plot_images_multienv"]

def _adjust_image(im):
    """
    Adjusts the image to be plotted.
    """
    if im.dtype == np.float32 or im.dtype == np.float64:
        # Normalize to [0, 1] range
        im_min, im_max = im.min(), im.max()
        if im_max > 1 or im_min < 0:
            im = (im - im_min) / (im_max - im_min)
    else:
        # Clip and convert to integers if needed
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im

def plot_multienv(dataset, filename, random=True, extension = ".png"):
    if type(random) == bool:
        iterate = np.random.randint(0, len(dataset), 2) if random else range(2)
    else: # random is an iterable
        iterate = random

    for i in iterate:
        multienv_dict = dataset.__getitem__(i)

        envs = list(multienv_dict.keys())
        fig, axs = plt.subplots(1, len(envs), figsize=(len(envs)*5, 5))
        for e in range(len(envs)):
            env = envs[e]

            # The normalization is with respect all the data, and it's not necessary.
            #im_tens = multienv_dict[env][0]
            #rescaled_im = (im_tens - im_tens.min()) / (im_tens.max() - im_tens)
            #im = np.transpose(rescaled_im.numpy(), (1, 2, 0))

            im = np.transpose(multienv_dict[env][0].numpy(), (1, 2, 0)) 
            lab = multienv_dict[env][1]
            axs[e].imshow(_adjust_image(im))
            axs[e].set_title(str(lab))
        plt.suptitle(f"Comparison: {envs}")

        savepath = filename + "_" + str(i) + extension
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        plt.savefig(savepath)


def plot_images_multienv(images: List, subtitles: List, filename, extension = ".png"):
    assert len(images) == len(subtitles), "The number of images and labels is not the same."

    envs = len(images)
    fig, axs = plt.subplots(1, envs, figsize=(envs*5, 5))
    for e in range(envs):
        im = np.transpose(images[e].numpy(), (1, 2, 0)) 
        axs[e].imshow(_adjust_image(im))
        axs[e].set_title(str(subtitles[e]))

    savepath = filename + extension
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    plt.savefig(savepath)

import pandas as pd
import torch
from pytorch_lightning import LightningModule

def get_pa_metrics(module: LightningModule):
    df = pd.read_csv(os.path.join(module.trainer.logger.log_dir, "metrics.csv"))
    beta_epoch = df["val/beta"].dropna().values
    logPA_epoch = df["val/logPA"].dropna().values
    return torch.tensor(beta_epoch), torch.tensor(logPA_epoch)


def get_acc_metrics(module: LightningModule):
    df = pd.read_csv(os.path.join(module.trainer.logger.log_dir, "metrics.csv"))
    AFR_pred = df["val/AFR pred"].dropna().values
    AFR_true = df["val/AFR true"].dropna().values
    acc_pa = df["val/acc_pa"].dropna().values

    assert len(AFR_pred) == len(AFR_true) == len(acc_pa), "Some AFR or acc_pa values are not being stored properly."
    assert len(AFR_pred) == 1, "There should be only one value for AFR and acc_pa."

    return AFR_pred[0], AFR_true[0], acc_pa[0]