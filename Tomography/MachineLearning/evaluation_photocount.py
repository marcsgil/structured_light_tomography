from juliacall import Main as jl
import h5py
import torch
import models
from torchvision.transforms import v2
import numpy as np
from training import fidelity

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

for order in range(1,5):
    print(order)
    with h5py.File('Data/Processed/pure_photocount.h5') as f:
        images = format_data(f[f'images_order{order}/2048_photocounts'][:].astype(np.float32))
        _lables = torch.from_numpy(f[f'labels_order{order}'][:50]).to(device)

    labels = torch.cat([torch.real(_lables),torch.imag(_lables)],dim=1)
                        
    
    for pc in [2**k for k in range(6,12)]:
        model = torch.load(f"Results/MachineLearningModels/Photocount/Order{order}/{pc}_photocounts/checkpoint.pt").to(device)

        with torch.no_grad():
            labels_pred = model(images)

        
        print(fidelity(labels_pred, labels).mean())