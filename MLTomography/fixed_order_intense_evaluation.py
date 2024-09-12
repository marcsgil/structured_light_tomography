from keras.models import load_model
import keras
import numpy as np
from h5py import File
from juliacall import Main as jl
jl.seval("using BayesianTomography")


def crop_indices(width, height, xmin, xmax, ymin, ymax, Xmin, Xmax, Ymin, Ymax):

    # Calculate the cropping coordinates
    lower = int(round(height * (Ymax - ymin) / (ymax - ymin)))
    upper = int(round(height * (Ymin - ymin) / (ymax - ymin)))
    left = int(round(width * (Xmin - xmin) / (xmax - xmin)))
    right = int(round(width * (Xmax - xmin) / (xmax - xmin)))

    return max(upper, 1), min(lower, height), max(left, 1), min(right, width)


with File('../Data/Processed/fixed_order_intense.h5') as f:
    direct_lims = f['direct_lims'][:]
    converted_lims = f['converted_lims'][:]

for order in range(1, 6):
    R = 2.5 + 0.5*order
    upper_d, lower_d, left_d, right_d = crop_indices(
        400, 400, *direct_lims, -R, R, -R, R)
    upper_c, lower_c, left_c, right_c = crop_indices(
        400, 400, *converted_lims, -R, R, -R, R)

    with File('../Data/Processed/fixed_order_intense.h5') as f:
        direct = keras.layers.Resizing(64, 64)(
            f[f'images_order{order}'][:, 0, upper_d:lower_d, left_d:right_d])
        converted = keras.layers.Resizing(64, 64)(
            f[f'images_order{order}'][:, 1, upper_c:lower_c, left_c:right_c])
        rho = np.conj(f[f'labels_order{order}'][:])

    x = np.stack((direct, converted), axis=1)
    mu = x.mean(axis=(-1, -2), keepdims=True)
    sigma = x.std(axis=(-1, -2), keepdims=True)
    x = (x - mu) / sigma

    model = load_model(f"TrainedModels/FixedOrderIntense/order{order}_trial1.keras")

    y_pred = np.array(model(x))

    fids = np.empty(rho.shape[0], dtype="float32")

    for n in np.arange(len(fids)):
        rho_pred = jl.density_matrix_reconstruction(y_pred[n])
        fids[n] = jl.fidelity(rho[n], rho_pred)

    print(fids.mean())
