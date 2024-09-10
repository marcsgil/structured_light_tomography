import models_new
import keras
import os
from data_loading import load_hdf5_dataset
import h5py

os.environ["KERAS_BACKEND"] = "jax"

num_classes = 3
input_shape = (2, 64, 64)

model = models_new.LeNet5(input_shape, num_classes)

# x, y = load_hdf5_dataset("../Data/Training/positive_l.h5", expand_dims_axis=1)

with h5py.File('../Data/Training/mixed_intense.h5', 'r') as f:
    x = f['images_order1'][:]
    y = f['labels_order1'][:]

mu = x.mean(axis=(-1, -2), keepdims=True)
sigma = x.std(axis=(-1, -2), keepdims=True)
x = (x - mu) / sigma

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
)

batch_size = 64
epochs = 200

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="TrainedModels/best_model.keras", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=40),
]

model.fit(
    x,
    y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
