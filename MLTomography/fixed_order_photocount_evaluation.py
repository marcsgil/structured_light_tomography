from keras.models import load_model
import keras
import numpy as np
import h5py
from juliacall import Main as jl
jl.seval("using BayesianTomography")

for order in range(4, 5):
    for pc in [2**i for i in range(6, 12)]:
        with h5py.File('../Data/Processed/pure_photocount.h5', 'r') as f:
            x = f[f'images_order{order}/{pc}_photocounts'][:]
            y = f[f'labels_order{order}/{pc}_photocounts'][:]

        mu = x.mean(axis=(-1, -2), keepdims=True)
        sigma = x.std(axis=(-1, -2), keepdims=True)
        x = (x - mu) / sigma

        model = load_model(f"TrainedModels/FixedOrderIntense/order{order}.keras")

        y_pred = np.array(model(x))

        fids = np.empty(rho.shape[0], dtype="float32")

        for n in np.arange(len(fids)):
            rho_pred = jl.density_matrix_reconstruction(y_pred[n])
            fids[n] = jl.fidelity(rho[n], rho_pred)

        print(fids.mean())
