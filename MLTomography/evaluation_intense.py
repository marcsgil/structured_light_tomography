import h5py
import torch
import models
from torchvision.transforms import v2
import numpy as np
import scipy.linalg as la

def X_matrix(j,k,d):
    X = np.zeros((d,d),dtype=complex)
    X[j,k] = 1
    X[k,j] = 1
    return X / la.norm(X)

def Y_matrix(j,k,d):
    Y = np.zeros((d,d),dtype=complex)
    Y[j,k] = -1j
    Y[k,j] = 1j
    return Y / la.norm(Y)

def Z_matrix(j,d):
    Z = np.zeros((d,d),dtype=complex)
    if j == -1:
        for k in range(d):
            Z[k,k] = 1
    else:
        for k in range(j+1):
            Z[k,k] = 1
        Z[j+1,j+1] = -j-1
    return Z / la.norm(Z)

def gell_man_matrices(d):
    X = [X_matrix(j,k,d) for j in range(d) for k in range(j)]
    Y = [Y_matrix(j,k,d) for j in range(d) for k in range(j)]
    Z = [Z_matrix(j,d) for j in range(d-1)]
    return np.concatenate([[Z_matrix(-1,d)],X,Y,Z])

def linear_combination(xs,basis):
    d = np.sqrt(len(xs) + 1)
    s = sum([x*b for x,b in zip(xs,basis[1:])])
    return basis[0] / np.sqrt(d) + s

def fidelity(rho, sigma):
    sqrt_rho = la.sqrtm(rho)
    return (np.real(np.trace(la.sqrtm(sqrt_rho @ sigma @ sqrt_rho))))**2

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

with h5py.File('../Data/Raw/positive_l.h5') as f:
    transform = v2.Compose([
        torch.from_numpy,
        v2.Resize((64, 64))])

    images = f['images_dim2'][:]
    images = np.expand_dims(images, 1)
    images = transform(images).float()
            
    def normalize(x):
        mean = [x.mean()]
        std = [x.std()]
        return v2.Normalize(mean=mean, std=std)(x)


    images_exp = normalize(images).to(device)
    labels_exp = f[f'images_dim2'].attrs["density_matrices"]
                    

model = torch.load(f"Test/checkpoint.pt").to(device)

with torch.no_grad():
    labels_pred = model(images_exp).cpu().numpy()

basis = gell_man_matrices(2)
sigmas = [linear_combination(rho, basis) for rho in labels_pred]

fidelities = np.array([fidelity(sigmas[n], labels_exp[n]) for n in range(100)])

print(fidelities.mean())
print(fidelities.std())