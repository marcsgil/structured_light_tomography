import numpy as np
from abc import (ABC, abstractmethod)
from dataclasses import dataclass
from quantum_samplers.samplers import sample_haar_vectors, sample_density_matrices
from multimethod import multimethod
from structured_light import fixed_order_basis
from einsumt import einsumt as einsum
import matplotlib.pyplot as plt
import torch


@dataclass
class FixedOrderDataset(ABC):
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
class PureDataset(FixedOrderDataset):
    @multimethod
    def generate_dataset(self, c):
        basis = self.get_basis()
        result = einsum('mk,knij->mnij', c, basis)
        return np.real(result*result.conj()), real_representation(c)

    @multimethod
    def generate_dataset(self, n_samples: int):
        c = sample_haar_vectors(n_samples, self.order+1)
        return self.generate_dataset(c)

    def translate_output(self, y, make_first_real=False):
        if make_first_real:
            cs = complex_representation(y)
            standardize(cs)
            return cs
        else:
            return complex_representation(y)


@dataclass
class MixedDataset(FixedOrderDataset):

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
