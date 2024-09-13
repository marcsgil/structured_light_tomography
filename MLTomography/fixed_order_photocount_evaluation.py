from keras.models import load_model
import keras
import numpy as np
import h5py
from juliacall import Main as jl
from loss import fidelity_loss
jl.seval("using BayesianTomography")
jl.seval('include("../Utils/pure_state_utils.jl")')


for order in range(1, 2):
    for pc in [2**i for i in range(6, 12)]:
        with h5py.File('../Data/Processed/pure_photocount.h5', 'r') as f:
            x = f[f'images_order{order}/{pc}_photocounts'][:]
            psi = f[f'labels_order{order}'][:50]

        mu = x.mean(axis=(-1, -2), keepdims=True)
        sigma = x.std(axis=(-1, -2), keepdims=True)
        x = (x - mu) / sigma

        for trial in range(1,2):
            model = load_model(
                f"TrainedModels/FixedOrderPhotocount/order{
                    order}_{pc}_photocounts_trial{trial}.keras",
                custom_objects={'fidelity_loss': fidelity_loss})

            y_pred = np.array(model(x))

            fids = np.empty(y_pred.shape[0], dtype="float32")

            for n in np.arange(len(fids)):
                psi_pred = jl.decat_real_and_imag(y_pred[n])
                fids[n] = jl.fidelity(psi[n], psi_pred)

            print(f'Order {order}; {pc} photocounts; trial {trial}:',  fids.mean())
