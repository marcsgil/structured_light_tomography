import numpy as np
import h5py
from juliacall import Main as jl
jl.seval("using BayesianTomography")
import torch
from torchvision.transforms import v2
import models

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def crop_indices(width, height, xmin, xmax, ymin, ymax, Xmin, Xmax, Ymin, Ymax):

    # Calculate the cropping coordinates
    lower = int(round(height * (Ymax - ymin) / (ymax - ymin)))
    upper = int(round(height * (Ymin - ymin) / (ymax - ymin)))
    left = int(round(width * (Xmin - xmin) / (xmax - xmin)))
    right = int(round(width * (Xmax - xmin) / (xmax - xmin)))

    return max(upper, 1), min(lower, height), max(left, 1), min(right, width)


with h5py.File('../Data/Processed/mixed_intense.h5') as f:
    direct_lims = f['direct_lims'][:]
    converted_lims = f['converted_lims'][:]
    images = f['images_order1'][:]

fidelities = np.empty(100)

for (k, order) in enumerate(range(1, 2)):
    with h5py.File('../Data/Processed/mixed_intense.h5') as f:
        direct_lims = f['direct_lims'][:]
        converted_lims = f['converted_lims'][:]

        R = 2.5 + 0.5*order
        upper_d, lower_d, left_d, right_d = crop_indices(
            400, 400, *direct_lims, -R, R, -R, R)
        upper_c, lower_c, left_c, right_c = crop_indices(
            400, 400, *converted_lims, -R, R, -R, R)

        transform = v2.Compose([
            torch.from_numpy,
            v2.Resize((64, 64))])

        direct = transform(
            f[f'images_order{order}'][:, 0, upper_d:lower_d, left_d:right_d]).float()
        converted = transform(
            f[f'images_order{order}'][:, 1, upper_c:lower_c, left_c:right_c]).float()

        def normalize(x):
            mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
            std = [x[:, n, :, :].std() for n in range(x.shape[1])]
            return v2.Normalize(mean=mean, std=std)(x)

        images_exp = normalize(torch.stack((direct, converted), 1)).to(device)
        labels_exp = np.conj(f[f'labels_order{order}'][:])

    model = torch.load(f"TrainedModels/checkpoint.pt").to(device)

    with torch.no_grad():
        labels_pred = model(images_exp).cpu().numpy()

    sigmas = [jl.density_matrix_reconstruction(rho) for rho in labels_pred]

    for n in range(100):
        fidelities[n] = jl.fidelity(sigmas[n], labels_exp[n])

print(fidelities.mean())
