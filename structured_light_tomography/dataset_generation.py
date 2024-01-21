import cv2
from noise import pnoise2
import numpy as np
from abc import (ABC, abstractmethod)
from dataclasses import dataclass
from quantum_samplers.samplers import sample_haar_vectors, sample_density_matrices
from multimethod import multimethod
from structured_light import fixed_order_basis
from einsumt import einsumt as einsum
import matplotlib.pyplot as plt
from structured_light_tomography.photocount_treatment import array_representation

import torch
from torchvision.transforms import v2


@dataclass
class FixedOrderModes(ABC):
    order: int
    grid_res: int
    rmax: float
    angle: float
    waist: float

    @abstractmethod
    def generate_dataset(self, n_samples):
        ...

    @abstractmethod
    def translate_output(self, y, *args):
        ...

    def get_basis(self):
        r = np.linspace(-self.rmax, self.rmax, self.grid_res)

        x, y = np.meshgrid(r, r)

        direct_basis = fixed_order_basis(
            x, y, self.waist, self.order).astype(np.complex64)

        astig_basis = direct_basis.copy()
        for n, _ in enumerate(astig_basis):
            astig_basis[n] *= np.exp(-1j * n * self.angle)

        return np.stack([direct_basis, astig_basis], axis=1)


def real_representation(c):
    return np.concatenate([np.real(c), np.imag(c)], axis=1)


def complex_representation(y):
    N = y.shape[1] // 2
    return y[:, :N] + 1j * y[:, N:]


def standardize(cs):
    for c in cs:
        c *= c[0].conj()/np.linalg.norm(c)


@dataclass
class PureModes(FixedOrderModes):
    @multimethod
    def generate_dataset(self, c, augment=False):
        basis = self.get_basis()
        result = einsum('mk,knij->mnij', c, basis)
        x = np.real(result*result.conj())
        if augment:
            x, y = augment_dataset(x, real_representation(c))
            return x.numpy(), y.numpy()
        else:
            return x, real_representation(c)

    @multimethod
    def generate_dataset(self, n_samples: int, augment=False):
        c = sample_haar_vectors(n_samples, self.order+1)
        return self.generate_dataset(c, augment)

    def translate_output(self, y, make_first_real=False):
        if make_first_real:
            cs = complex_representation(y)
            standardize(cs)
            return cs
        else:
            return complex_representation(y)


@dataclass
class MixedModes(FixedOrderModes):

    @multimethod
    def generate_dataset(self, rhos):
        n_samples = rhos.shape[0]
        basis = self.get_basis()

        x = np.real(einsum(
            'imn,mjkl,njkl->ijkl', rhos, basis, basis.conj()))

        dim = self.order + 1
        A = np.empty((n_samples, dim))
        J, K = np.tril_indices(dim, -1)
        B = np.empty((n_samples, len(J)))
        C = np.empty((n_samples, len(J)))

        sqrt2 = np.sqrt(2)

        for n, rho in enumerate(rhos):
            A[n, :] = np.real(np.diag(rho))

            for j in range(len(J)):
                entry = rho[J[j], K[j]]
                B[n, j] = np.real(entry) * sqrt2
                C[n, j] = np.imag(entry) * sqrt2

        y = np.concatenate([A, B, C], axis=1)

        return x, y

    @multimethod
    def generate_dataset(self, n_samples: int):
        rhos = sample_density_matrices(n_samples, self.order + 1)
        return self.generate_dataset(rhos)

    def translate_output(self, y, project=False):
        # TODO: Implement projection
        n_samples = y.shape[0]
        dim = int(np.sqrt(y.shape[1]))
        slice_dim = dim*(dim-1)//2
        A = y[:, :dim]
        B = y[:, dim:dim+slice_dim]
        C = y[:, dim+slice_dim:]

        sqrt2 = np.sqrt(2)
        rhos = np.empty((n_samples, dim, dim), dtype=np.complex64)

        J, K = np.tril_indices(dim, -1)

        for n in range(n_samples):
            for i in range(dim):
                rhos[n, i, i] = A[n, i]

            for m, (j, k) in enumerate(zip(J, K)):
                entry = (B[n, m] + 1j * C[n, m]) / sqrt2
                rhos[n, j, k] = entry
                rhos[n, k, j] = np.conj(entry)

        return rhos


def generate_perlin_noise_2d(shape, scale, strength):
    I = 10 * np.random.randn()
    J = 10 * np.random.randn()
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i, j] = pnoise2(I + i / scale, J + j / scale)

    noise = (noise - np.mean(noise)) / np.std(noise) * strength + 1
    return noise


def random_pad(img, max_ratio):
    max_pad = int(img.shape[1] * max_ratio)
    h_or_v = np.random.randint(2)
    pad = np.random.randint(max_pad)

    if h_or_v == 0:
        result = np.pad(img, ((0, 0), (pad//2, pad//2)), 'minimum')
    else:
        result = np.pad(img, ((pad//2, pad//2), (0, 0)), 'minimum')

    max = np.max(result)
    result *= 255.0 / max

    return cv2.resize(result.astype(np.uint8), (img.shape[1], img.shape[0])) * max/255


def random_roll(img, max_ratio):
    max_roll = int(img.shape[1] * max_ratio)
    roll = np.random.randint(-max_roll, max_roll)

    return np.roll(img, roll, axis=1)


def augment_dataset(x, y,
                    alpha=50.0,
                    sigma=5.0,
                    pad_ratio=0.2,
                    roll_ratio=0.1,
                    perlin_scale=15,
                    perlin_strength=0.3):
    mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
    std = [x[:, n, :, :].std() for n in range(x.shape[1])]

    for img in x:
        for n in range(2):
            img[n] = random_roll(random_pad(img[n], pad_ratio), roll_ratio)
            img[n] *= generate_perlin_noise_2d(img[n].shape,
                                               perlin_scale, perlin_strength)

    X = v2.Compose([
        torch.from_numpy,
        v2.ElasticTransform(alpha, sigma),
        v2.Normalize(mean=mean, std=std),
    ])(x)
    Y = torch.from_numpy(y)
    return X, Y


def sample_photons(x, n_samples):
    for image in x:
        image[0] -= image[0].min()
        image[1] -= image[1].min()
        N = np.sum(image, axis=(1, 2))
        image[0] /= N[0]
        image[1] /= N[1]
        probabilities = image.flatten() / image.sum()
        history = np.random.choice(
            np.arange(len(probabilities)), n_samples, p=probabilities)
        image[:, :, :] = array_representation(history, image.shape)

    mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
    std = [x[:, n, :, :].std() for n in range(x.shape[1])]

    return v2.Normalize(mean=mean, std=std)(torch.from_numpy(x)).numpy()
