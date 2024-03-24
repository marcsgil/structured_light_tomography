from juliacall import Main as jl
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

def crop_indices(width, height, xmin, xmax, ymin, ymax, Xmin, Xmax, Ymin, Ymax):   
   
    # Calculate the cropping coordinates
    lower = int(round(height * (Ymax - ymin) / (ymax - ymin)))
    upper = int(round(height * (Ymin - ymin) / (ymax - ymin)))
    left = int(round(width * (Xmin - xmin) / (xmax - xmin)))
    right = int(round(width * (Xmax - xmin) / (xmax - xmin)))
    
    return max(upper, 1),min(lower,height),max(left,1),min(right,width)

with h5py.File('Data/Processed/mixed_intense.h5') as f:
    direct_lims = f['direct_lims'][:]
    converted_lims = f['converted_lims'][:]
    images = f['images_order1'][:]

fidelities = np.zeros((5,100))

for (k, order) in enumerate(range(1,6)):
    with h5py.File('Data/Processed/mixed_intense.h5') as f:
        direct_lims = f['direct_lims'][:]
        converted_lims = f['converted_lims'][:]

        R = 2.5 + 0.5*order
        upper_d,lower_d,left_d,right_d = crop_indices(400,400, *direct_lims, -R,R,-R,R)
        upper_c,lower_c,left_c,right_c = crop_indices(400,400, *converted_lims, -R,R,-R,R)

        transform = v2.Compose([
            torch.from_numpy,
            v2.Resize((64, 64))])

        direct = transform(f[f'images_order{order}'][:,0,upper_d:lower_d,left_d:right_d]).float()
        converted = transform(f[f'images_order{order}'][:,1,upper_c:lower_c,left_c:right_c]).float()
                
        def normalize(x):
            mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
            std = [x[:, n, :, :].std() for n in range(x.shape[1])]
            return v2.Normalize(mean=mean, std=std)(x)


        images_exp = normalize(torch.stack((direct,converted),1)).to(device)
        labels_exp = f[f'labels_order{order}'][:]
                        

    model = torch.load(f"Results/MachineLearningModels/Intense/Order{order}/checkpoint.pt").to(device)

    with torch.no_grad():
        labels_pred = model(images_exp).cpu().numpy()

    basis = gell_man_matrices(order+1)
    sigmas = [linear_combination(rho, basis) for rho in labels_pred]

    fidelities[k] = np.array([fidelity(sigmas[n], labels_exp[n]) for n in range(100)])

print(fidelities.mean(axis=1))
print(fidelities.std(axis=1))

with h5py.File('Results/Intense/New/machine_learning.h5', 'w-') as f:
    f["fids"] = fidelities.mean(axis=1)
    f["fids_std"] = fidelities.std(axis=1)