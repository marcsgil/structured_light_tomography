from slmcontrol import generate_hologram
from partially_coherent_sources import generate_masks
from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.linalg import eigh
from tqdm import trange
import numpy as np
from scipy.optimize import curve_fit
from slmcontrol import lg

def surface_fit(func, x, y, z, p0):
    # Flatten the x, y, and z arrays for curve_fit
    xdata = np.vstack((x.ravel(), y.ravel()))
    zdata = z.ravel()

    # Fit the function to the data
    return curve_fit(func, xdata, zdata, p0)

@dataclass
class Config:
    incoming: ArrayLike
    x: ArrayLike
    y: ArrayLike
    max_modulation: int
    x_period: int
    y_period: int

def loop_capture(desireds, camera, slm, config, sleep_time=0.15):
    incoming = config.incoming
    x = config.x
    y = config.y
    max_modulation = config.max_modulation
    x_period = config.x_period
    y_period = config.y_period
    
    N = desireds.shape[0]

    holo = generate_hologram(desireds[0], incoming, x, y, max_modulation, x_period, y_period, 0, 0)
    slm.updateArray(holo)

    result = camera.capture().astype(np.int64)

    for n in trange(1,N):
        holo = generate_hologram(desireds[n], incoming, x, y, max_modulation, x_period, y_period, 0, 0)
        result += camera.capture()
        slm.updateArray(holo, sleep_time)
    
    slm.updateArray(holo)
    result += camera.capture()
    return (result / N).round().astype(np.uint8)

def calculate_eigenstates(vectors, basis):
    r_basis = basis.reshape((basis.shape[0], -1))

    return np.matmul(vectors.T, r_basis).reshape(*basis.shape)

def basis_loop(mean_imgs, n_masks, basis, rhos,
    camera, slm, config, sleep_time=0.15):
    for (n, rho) in enumerate(rhos):
        vals, vecs = eigh(rho)
        eigen_states = calculate_eigenstates(vecs, basis)

        desireds = generate_masks(n_masks, vals, eigen_states)
        print("Capturing mode ", n+1)
        mean_imgs[n] = loop_capture(desireds, camera, slm, config, sleep_time)

        del eigen_states
        del desireds

def finite_diff_x_derivative(img):
    # Convert the input image to a NumPy array if it isn't already
    img = np.asarray(img)
    
    # Compute the finite differences along the x-axis
    x_derivative = img[1:, :] - img[:-1, :]
    
    return x_derivative

def finite_diff_r_derivative(img, x, y, x0, y0):
    
    # Initialize the r_derivative matrix
    r_derivative = np.empty((img.shape[0] - 1, img.shape[1] - 1), dtype=np.float64)
    
    for n in range(r_derivative.shape[1]):
        for m in range(r_derivative.shape[0]):
            dx = x[m] - x0
            dy = y[n] - y0

            r = np.sqrt(dx[m]**2 + dy[n]**2)
            r_derivative[m, n] = (dx[m] * (int(img[m+1, n]) - int(img[m, n])) +
                                  dy[n] * (int(img[m, n+1]) - int(img[m, n]))) / r
    
    return r_derivative

def laguerre_iris(X,w,x0,y0,radius,amplitude,background):
    x,y = X
    return amplitude * np.abs(lg(x-x0,y-y0,0,2,w))**2 * ( (x-x0)**2 + (y-y0)**2 < radius**2 ) + background

def gaussian_model(X, x0, y0, w, amplitude, background):
    x, y = X
    return amplitude * np.exp(-2 * ((x - x0)**2 + (y - y0)**2) / w**2) + background