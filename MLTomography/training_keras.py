import MLTomography.models_keras as models_keras
import keras.layers as layers
import keras
import numpy as np
import os
from data_loading import load_hdf5_dataset

os.environ["KERAS_BACKEND"] = "jax"

num_classes = 3
input_shape = (1, 64, 64)

model = models_keras.LeNet5(input_shape, num_classes)

x, y = load_hdf5_dataset("../Data/Training/positive_l.h5", expand_dims_axis=1)

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

batch_size = 128
epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="TrainedModels/best_model.keras", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
]

model.fit(
    x,
    y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)