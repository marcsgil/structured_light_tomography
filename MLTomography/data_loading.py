from h5py import File
import keras

def load_hdf5_dataset(path, x_key="images", y_key="labels", expand_dims_axis=None):
    with File(path) as file:
        x = file[x_key][:]
        y = file[y_key][:]

    N = x.sum(axis=(-1, -2), keepdims=True)
    x = x / N

    if expand_dims_axis is not None:
        x = keras.ops.expand_dims(x, expand_dims_axis)

    return x, y