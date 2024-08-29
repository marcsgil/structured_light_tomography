import keras_cv
import keras.layers as layers
import keras
import os

os.environ["KERAS_BACKEND"] = "jax"

model = keras_cv.models.ResNetV2Backbone(
    stackwise_filters=[64],
    stackwise_blocks=[2],
    stackwise_strides=[1],
    include_rescaling=False,
    input_shape=(64, 64, 1),
)

model.summary()
