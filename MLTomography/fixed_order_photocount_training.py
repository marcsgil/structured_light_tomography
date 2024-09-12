import models
import keras
import h5py
import os

input_shape = (2, 64, 64)
batch_size = 64
epochs = 300


def dot(x, y):
    return keras.layers.Dot(axes=1)((x, y))[:, 0]


def fidelity_loss(y1, y2):
    N = y1.shape[1] // 2
    r1 = y1[:, :N]
    i1 = y1[:, N:]
    r2 = y2[:, :N]
    i2 = y2[:, N:]

    R = (r1*r2 + i1*i2).sum(axis=1)
    I = (r1*i2 - r2*i1).sum(axis=1)

    return 1 - (R**2 + I**2) / (dot(y1, y1) * dot(y2, y2))


for order in range(1, 5):
    for pc in [2**i for i in range(6, 12)]:
        print(f"Training model for order {order} and {pc} photocounts")
        with h5py.File('../Data/Training/fixed_order_photocount.h5', 'r') as f:
            x = f[f'images_order{order}/{pc}_photocounts'][:]
            y = f[f'labels_order{order}/{pc}_photocounts'][:]

        mu = x.mean(axis=(-1, -2), keepdims=True)
        sigma = x.std(axis=(-1, -2), keepdims=True)
        x = (x - mu) / sigma

        num_classes = 2 * (order + 1)
        model = models.DefaultConvNet(input_shape, num_classes)
        model.compile(
            loss=fidelity_loss,
            optimizer=keras.optimizers.Adam(amsgrad=True),
        )

        logger_path = f"./logs/FixedOrderPhotocount/order{order}"
        os.makedirs(os.path.dirname(logger_path), exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"TrainedModels/FixedOrderPhotocount/order{order}/{pc}_photocounts.keras", save_best_only=True),
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
