import keras
from keras import layers

def ConvNet(input_shape, num_classes, conv_channels, kernel_size, activation, connected_channels):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(conv_channels[0], kernel_size=kernel_size, activation=activation)(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for channels in conv_channels[1:]:
        x = layers.Conv2D(channels, kernel_size=kernel_size, activation=activation)(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Flatten()(x)

    for channels in connected_channels:
        x = layers.Dense(channels, activation=activation)(x)

    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="ConvNet")

def DefaultConvNet(input_shape, num_classes):
    return ConvNet(input_shape, num_classes, [24,40,35], (5,5), 'elu', [120,80,40])