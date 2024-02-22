import numpy as np
from scipy.optimize import minimize


def format_image(image):
    image1, image2 = np.hsplit(image, 2)
    return np.stack((image2, image1))


def format_images(images):
    return np.array([format_image(image) for image in images])


def gaussian(extrema, shape):
    xmin, ymin, xmax, ymax = extrema
    x = np.linspace(xmin, xmax, shape[1])
    y = np.linspace(ymin, ymax, shape[0])
    X, Y = np.meshgrid(x, y)
    return np.exp(-X**2-Y**2)


def fit_grid(image):
    def loss(extrema, image):
        return np.sum((image - gaussian(extrema, image.shape))**2)

    norm_image = image / np.max(image)
    return minimize(loss, [-2.0, -2.0, 2.0, 2.0], norm_image, tol=1e-4)


def floor_relu(x):
    return np.uint8(np.maximum(0, np.floor(x)))


def extract_photons(frames, mean, std, threshold):
    return floor_relu((frames - mean) / threshold / std)


def circular_mask(images, radius, xd, yd, xc, yc):
    Xd, Yd = np.meshgrid(xd, yd)
    Xc, Yc = np.meshgrid(xc, yc)

    images[0][Xd**2 + Yd**2 > radius] = 0
    images[1][Xc**2 + Yc**2 > radius] = 0


def circular_masks(images, radius, xd, yd, xc, yc):
    for image in images:
        circular_mask(image, radius, xd, yd, xc, yc)


def history_vector(images):
    history = []
    for image in images:
        flattened = image.flatten()
        coords = np.argwhere(flattened > 0)
        for coord in coords:
            for _ in range(flattened[coord[0]]):
                history.append(coord[0])
    return np.array(history)


def array_representation(history, shape):
    array = np.zeros(shape, dtype=np.uint8)
    for event in history:
        idx = np.unravel_index(event, shape)
        array[idx] += 1
    return array
