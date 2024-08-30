from h5py import File
import keras

with File("../Data/Raw/positive_l.h5") as file:
    x = file["images_dim2"][:].astype("float32")

# x -= 2
N = x.sum(axis=(-1, -2), keepdims=True)
x = x / N
x = keras.ops.expand_dims(x, 1)

from keras.models import load_model

model = load_model("TrainedModels/best_model.keras")

thetas = model(keras.layers.Resizing(64, 64)(x))

with File("../Results/Intense/positive_l_ml.h5", "w-") as file:
    file["thetas"] = thetas