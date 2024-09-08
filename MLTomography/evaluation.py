from juliacall import Main as jl
jl.seval("using BayesianTomography")
from h5py import File
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.models import load_model

with File("../Data/Raw/positive_l.h5") as file:
    obj = file["images_dim2"]

    x = obj[:].astype("float32")
    rho = obj.attrs["density_matrices"]

x = np.expand_dims(x, 1)

x -= 2
N = x.sum(axis=(-1, -2), keepdims=True)
x = x / N


model = load_model("TrainedModels/best_model.keras")

y_pred = np.array(model(keras.layers.Resizing(64, 64)(x)))

fids = np.empty(rho.shape[0], dtype="float32")

for n in np.arange(len(fids)):
    rho_pred = jl.density_matrix_reconstruction(y_pred[n])
    fids[n] = jl.fidelity(rho[n], rho_pred)

print(fids.mean())