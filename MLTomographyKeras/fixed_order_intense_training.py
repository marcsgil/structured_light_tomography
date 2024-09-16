import models
import keras
import h5py
import os

input_shape = (2, 64, 64)
batch_size = 64
epochs = 300


def apply_layers(x, layers, batch_size):
    num_batches = x.shape[0] // batch_size

    for i in range(num_batches):
        for layer in layers:
            x[i*batch_size:(i+1)*batch_size] = layer(x[i *
                                                       batch_size:(i+1)*batch_size])

    x[num_batches*batch_size:x.shape[0]
      ] = layer(x[num_batches*batch_size:x.shape[0]])


layers = [
    keras.layers.RandomTranslation(0.2, 0.2, fill_mode='constant'),
    keras.layers.RandomZoom((-0.4, 0.1), (-0.4, 0.1), fill_mode='constant'),
]

for order in range(5, 6):
    #x = None

    with h5py.File('../Data/Training/fixed_order_intense.h5', 'r') as f:
        x = f[f'images_order{order}'][:]
        y = f[f'labels_order{order}'][:]

    #print("Applying augmentations")
    #apply_layers(x, layers, 512)

    mu = x.mean(axis=(-1, -2), keepdims=True)
    sigma = x.std(axis=(-1, -2), keepdims=True)
    x = (x - mu) / sigma

    num_classes = (order + 1)**2 - 1

    for trial in range(1, 2):
        print(f"Training model for order {order}; trial {trial}")
        model = models.DefaultConvNet(input_shape, num_classes)
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(amsgrad=True),
        )

        logger_path = f"./logs/FixedOrderIntense/order{order}_trial{trial}.csv"
        os.makedirs(os.path.dirname(logger_path), exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"TrainedModels/FixedOrderIntense/order{order}_trial{trial}.keras", save_best_only=True),
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