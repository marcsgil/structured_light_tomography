import models
import keras
import h5py
from loss import fidelity_loss
import os

input_shape = (2, 64, 64)
batch_size = 256
epochs = 300


for order in range(2, 5):
    for pc in [2**i for i in range(6, 12)]:
        with h5py.File('../Data/Training/fixed_order_photocount.h5', 'r') as f:
            x = f[f'images_order{order}/{pc}_photocounts'][:]
            y = f[f'labels_order{order}/{pc}_photocounts'][:]

        mu = x.mean(axis=(-1, -2), keepdims=True)
        sigma = x.std(axis=(-1, -2), keepdims=True)
        x = (x - mu) / sigma

        num_classes = 2 * (order + 1)

        for trial in range(1, 2):
            print(f"Training model for order {order}; {pc} photocounts; trial {trial}")
            model = models.DefaultConvNet(input_shape, num_classes)
            model.compile(
                loss=fidelity_loss,
                optimizer=keras.optimizers.Adam(amsgrad=True),
            )

            logger_path = f"./logs/FixedOrderPhotocount/order{
                order}_{pc}_photocounts_trial{trial}.csv"
            os.makedirs(os.path.dirname(logger_path), exist_ok=True)

            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f"TrainedModels/FixedOrderPhotocount/order{order}_{pc}_photocounts_trial{trial}.keras", save_best_only=True),
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
