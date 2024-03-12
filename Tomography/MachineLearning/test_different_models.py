import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import models
import training
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plts
from os.path import join
import torchvision
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import random_split

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def format_data(x):
    mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
    std = [x[:, n, :, :].std() for n in range(x.shape[1])]

    X = v2.Compose([
            torch.from_numpy,
            v2.Normalize(mean=mean, std=std),
            v2.Resize((64, 64)),
        ])(x).to(device)
    return X