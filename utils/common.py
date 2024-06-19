import numpy as np
import numba as nb

@nb.njit("f8(f8, f8, f8)")
def laplace_gamma(u, a, b):
    """
    Laplace transform of a Gamma random variable with
    shape parameter 'a' and rate parameter 'b' evaluated at 'u'
    """
    return b**a / (u + b)**a


@nb.njit("f8[:, :](f8[:], f8[:], f8[:])")
def uninorm_lpdf_many2many(data, means, vars):
    """
    Evaluates the log probability density of 'n' observations (data)
    for each of 'm' choices of mean and variance. Returns a n x m matrix
    """
    return -0.5 * (np.expand_dims(data, -1) - means)**2 / vars - \
        np.log(np.sqrt(2 * np.pi * vars))

@nb.njit("f8(f8[:], f8, f8)")
def uninorm_lpdf_many2one(data, mean, var):
    """
    Evaluates the log probability density of 'n' observations (data)
    from a gaussian distribution (returns the sum).
    """
    return np.sum(-0.5 * (data - mean)**2 / var - \
        np.log(np.sqrt(2 * np.pi * var)))


@nb.njit("f8[:](f8[:],f8[:],f8[:],f8[:])")
def eval_mixture_density(xgrid, means, vars, weights):
    dens_in_comp = np.exp(uninorm_lpdf_many2many(xgrid, means, vars))
    return np.sum(dens_in_comp * weights, axis=1)


@nb.jit("f8[:](f8[:])")
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x /np.sum(e_x)


@nb.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
