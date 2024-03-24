import h5py
import torch
import models
from torchvision.transforms import v2
import numpy as np
from training import fidelity
import time

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

fidelities = np.zeros((4,6,50))

T0 = time.time()
for (k, order) in enumerate(range(1,5)):
    with h5py.File('Data/Processed/pure_photocount.h5') as f:
        images = format_data(f[f'images_order{order}/2048_photocounts'][:].astype(np.float32))
        _lables = torch.from_numpy(f[f'labels_order{order}'][:50]).to(device)

    labels = torch.cat([torch.real(_lables),torch.imag(_lables)],dim=1)
                        
    
    for (j,pc) in enumerate([2**k for k in range(6,12)]):
        model = torch.load(f"Results/MachineLearningModels/Batch5/Photocount/Order{order}/{pc}_photocounts/checkpoint.pt").to(device)

        with torch.no_grad():
            labels_pred = model(images)

        fidelities[k,j] = fidelity(labels_pred, labels).cpu().numpy()

    t0 = time.time()
    model(images[:0])
    t1 = time.time()

    print(f"Time for one image: {t1-t0}")

T1 = time.time()
print(f"Total time: {T1-T0}")
        

"""with h5py.File('Results/Photocount/New/machine_learning.h5', 'w') as f:
    f["fids"] = fidelities.mean(axis=2)
    f["fids_std"] = fidelities.std(axis=2)"""