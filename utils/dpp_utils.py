import numpy as np
import numba as nb
import numpy.core.numeric as nx
from math import gamma
from utils.common import laplace_gamma

K_RANGE = np.arange(-25, 25 + 1)


@nb.njit("f8[:](f8, f8, f8)")
def power_exp_phis(rho, nu, s):
    """
    Computes the spectral density of the power exponential DPP (See
    Eq (3.22) in Lavancier et al [https://arxiv.org/pdf/1205.4818])
    for alpha = s alpha_max, s in (0, 1).
    """
    amax = np.sqrt(np.pi) * gamma(1.0 / nu + 1.0) / (rho * gamma(1.5))
    alpha = s * amax
    phis = s * np.exp(- np.abs(alpha * K_RANGE)**nu  )
    return phis


@nb.njit("f8[:, :](f8[:], f8[:], f8[:])")
def compute_c_app(points, phitildes, R):
    """
    Approximates the kernel driving the DPP density C(x, y) for (x, y) in 'points'.
    """
    points = (points - np.mean(R)) / (R[1] - R[0])
    n = len(points)
    c_app = np.empty((n, n), dtype=np.float64)
    diag_elem = np.sum(phitildes)
    for i in range(n):
        c_app[i, i] = diag_elem
        for j in range(i):
            c_app[i, j] = np.sum(phitildes * np.cos(2 * np.pi * K_RANGE * (points[i] - points[j])))
            c_app[j, i] = c_app[i, j]

    return c_app


@nb.njit("f8[:, :](f8[:], f8[:])")
def get_normalized_diffs(points, R):
    points = (points - np.mean(R)) / (R[1] - R[0])
    diffs = points - np.expand_dims(points, -1)
    return diffs 


def dpp_density_(points, phitildes, R, logscale=True, propto=True):
    """
    Computes the density of a DPP over R
    """
    c_app = compute_c_app(points, phitildes, R)
    out = np.log(np.linalg.det(c_app))
    if not propto:
        d_app = np.sum(np.log(1 + phitildes))
        out += (R[1] - R[0]) - d_app
        out -= len(points) * np.log(R[1] - R[0])
    if not logscale:
        out = np.exp(out)
    return out


@nb.jit("f8(f8, f8[:], f8[:], f8[:], f8[:], f8, f8, f8)")
def birth_arate_(new_loc, curr_means, alloc_means, phitildes, R, u, jump_a, jump_b):
    """
    Computes the acceptance rate of a birth proposal adding one atom 'new_loc' to the
    current state
    """
    m = len(alloc_means) + len(curr_means)
    all_atoms = np.concatenate((alloc_means, curr_means, np.array([new_loc])))
    c_app = compute_c_app(all_atoms, phitildes, R)
    # get matrix blocks
    A = c_app[:(m-1), :(m-1)]
    dens_ratio = laplace_gamma(u, jump_a, jump_b) * np.linalg.det(c_app) / np.linalg.det(A)
    out = dens_ratio / (len(curr_means) + 1)
    return out