import numpy as np
import os
import h5py

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras.layers as layers
import models

num_classes = 3
input_shape = (1, 64, 64)

model = models.create_cnn_regression_model(input_shape, num_classes)

with h5py.File("../../Data/Training/positive_l.h5") as file:
    x_train = file["images"][:]
    y_train = file["labels"][:]

    #x_test = file["images"][85000:10**5]
    #y_test = file["labels"][85000:10**5]

x_train = np.expand_dims(x_train, 1)

model.compile(
    loss=keras.losses.MeanSquaredError,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

batch_size = 128
epochs = 5

callbacks = [
    #keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    #keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)