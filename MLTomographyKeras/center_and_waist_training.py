import models
import keras
import h5py
import os
import numpy as np

input_shape = (1, 64, 64)
batch_size = 64
epochs = 300

with h5py.File('../Data/Training/center_and_waist.h5') as f:
    x = f['x'][:]
    y = f['y'][:]

mu = x.mean(axis=(-1, -2), keepdims=True)
sigma = x.std(axis=(-1, -2), keepdims=True)
x = (x - mu) / sigma
x = np.expand_dims(x, axis=1)

num_classes = 3
model = models.DefaultConvNet(input_shape, num_classes)
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(amsgrad=True),
)

logger_path = f"./logs/center_and_waist.csv"
os.makedirs(os.path.dirname(logger_path), exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=f"TrainedModels/center_and_waist.keras", save_best_only=True),
    keras.callbacks.EarlyStopping(patience=40),
    keras.callbacks.CSVLogger(logger_path),
]

model.fit(
    x,
    y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
