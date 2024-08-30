import keras
from keras import layers
from keras import ops

def LeNet5(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional layer
    x = layers.Conv2D(6, kernel_size=(5, 5), activation='tanh')(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Second convolutional layer
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='tanh')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Flatten layer
    x = layers.Flatten()(x)
    
    # Fully connected layers
    x = layers.Dense(120, activation='tanh')(x)
    x = layers.Dense(84, activation='tanh')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes)(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="LeNet-5")
    
    return model


def base_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution block
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution block
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    return keras.Model(inputs, x)


def standard_model(input_shape, num_classes):
    model = base_model(input_shape)

    inputs = keras.Input(shape=input_shape)
    x = model(inputs)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)


def heuristic_model(input_shape, num_classes):
    model = base_model(input_shape)

    inputs = keras.Input(shape=input_shape)
    x = model(inputs)

    L = input_shape[-1] * input_shape[-2]

    T = layers.Dense(num_classes * L)(x)
    T = layers.Reshape((num_classes, L))(T)
    r_input = layers.Reshape((L,))(inputs)
    b = layers.Dense(num_classes)(x)

    outputs = keras.ops.einsum('nab,nb->na', T, r_input)
    outputs += b

    model = keras.Model(inputs, outputs)
    return model


def create_cnn_regression_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution block
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Second convolution block
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Third convolution block
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes)(x)
    
    model = keras.Model(inputs, outputs)
    return model


def create_cnn_regression_model2(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution block
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution block
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    L = input_shape[-1] * input_shape[-2]

    T = layers.Dense(num_classes * L)(x)
    T = layers.Reshape((num_classes, L))(T)
    r_input = layers.Reshape((L,))(inputs)
    b = layers.Dense(num_classes)(x)

    outputs = keras.ops.einsum('nab,nb->na', T, r_input)
    outputs += b

    model = keras.Model(inputs, outputs)
    return model
