import models
import keras
import h5py

input_shape = (2, 64, 64)
batch_size = 64
epochs = 300

for order in range(1, 6):
    print(f"Training model for order {order}")
    with h5py.File('../Data/Training/mixed_intense.h5', 'r') as f:
        x = f[f'images_order{order}'][:]
        y = f[f'labels_order{order}'][:]

    mu = x.mean(axis=(-1, -2), keepdims=True)
    sigma = x.std(axis=(-1, -2), keepdims=True)
    x = (x - mu) / sigma

    num_classes = (order + 1)**2 - 1
    model = models.DefaultConvNet(input_shape, num_classes)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(amsgrad=True),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"TrainedModels/FixedOrderIntense/order{order}.keras", save_best_only=True),
        keras.callbacks.EarlyStopping(patience=40),
        keras.callbacks.TensorBoard(
            log_dir=f"./logs/FixedOrderIntense/order{order}"),
    ]

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
    )
