import numpy as np
from copy import deepcopy
from tensorflow_probability.substrates import numpy as tfp
from dpp_state import State, Prior
from sklearn.metrics import pairwise_distances
from math import isfinite
from scipy.special import logit, expit
tfd = tfp.distributions
import numba as nb
import numpy.core.numeric as nx

from utils.dpp_utils import *
from utils.common import *


K_RANGE = np.arange(-25, 25 + 1)


def compute_phis(state):
    """
    Adds to the state the evaluation of the phi(k) and 
    phi(k) / (1 - phi(k)) for k in K_RANGE
    """
    state.phis = power_exp_phis(state.rho, state.nu, state.s)
    state.phitildes = state.phis / (1.0 - state.phis)
    return state


def dpp_density(state, prior, logscale=True, propto=True):
    """
    Computes the density of the DPP at the current atoms
    """
    all_atoms = np.concatenate([state.alloc_means, state.non_alloc_means])
    return dpp_density_(all_atoms, state.phitildes, prior.R, logscale, propto)


def sample_mean(atom_idx, curr_data, state, prior):
    """
    Metropolis-Hastings step to sample from the full-conditional distribution of
    the 'atom_idx' allocated mean. 
    
    Parameters
    ----------
    - atom_idx: index of the atom to sample
    - curr_data: vector of the observations allocated to atom_idx
    - state: instance of DPPState
    - prior: instance of DPPprior
    """
    curr_val = state.alloc_means[atom_idx]
    var = state.alloc_vars[atom_idx]
    curr_ll = uninorm_lpdf_many2one(curr_data, curr_val, var)
    curr_prior = dpp_density(state, prior)
    
    if np.random.uniform() < 0.2:
        prop_val = np.random.uniform(prior.R[0], prior.R[1])
    else:
        prop_val = np.random.normal(0, prior.mh_sigma) + curr_val
    state.alloc_means[atom_idx] = prop_val

    prop_ll = uninorm_lpdf_many2one(curr_data, prop_val, var)
    prop_prior = dpp_density(state, prior)

    assert(isfinite(prop_ll))
    assert(isfinite(prop_prior))
    assert(isfinite(curr_ll))
    assert(isfinite(curr_prior))

    arate = prop_ll + prop_prior - (curr_ll + curr_prior)
    if np.log(np.random.uniform()) < arate:
        state.alloc_means[atom_idx] = prop_val
    else: 
        state.alloc_means[atom_idx] = curr_val

    return state


@nb.jit("f8(f8[:], f8, f8, f8)")
def sample_var(data_in_clus, mean, prior_a, prior_b):
    """
    Samples from the full conditional distribution of the variance parameter
    """
    n = len(data_in_clus)
    post_a = prior_a + n / 2
    post_b = prior_b + 0.5 * np.sum((data_in_clus - mean)**2)
    return 1.0 / np.random.gamma(post_a, 1.0/ post_b)


def update_alloc_atoms(data, state, prior):
    """
    Cycle through the allocated atoms and update each of them via a Metropolis
    Hastings step.
    """
    data_by_clus = [data[state.clus == i] for i in np.unique(state.clus)]
    for i in range(len(state.alloc_means)):
        state = sample_mean(i, data_by_clus[i], state, prior)
        state.alloc_vars[i] = sample_var(
            data_by_clus[i], state.alloc_means[i], prior.var_a, prior.var_b)
    return state


def update_alloc_jumps(state, prior):
    """
    Sample from the full conditional distribution of the allocated jumps
    """
    cluscount = np.sum(state.clus[:, np.newaxis] == 
                       np.arange(state.alloc_jumps.shape[0]), axis=0)
    state.alloc_jumps = np.random.gamma(
        cluscount + prior.jump_a, 1.0/(prior.jump_b + state.u))
    return state


def update_non_alloc_atoms(state, prior):
    """
    Samples the non allocated atoms via a birth-death Metropolis Hastings move
    """
    def birt_prop():
        new_loc = np.random.uniform(prior.R[0], prior.R[1])
        return new_loc, state.non_alloc_means
    
    def death_prop():
        k = len(state.non_alloc_means)
        if k == 0:
            return None, None, None
        idx = np.random.choice(np.arange(len(state.non_alloc_means)))
        del_loc = state.non_alloc_means[idx]
        tmp = np.copy(state.non_alloc_means)
        return del_loc, np.delete(tmp, idx), idx
    
    def birth_arate(new_loc, curr_means):
        return birth_arate_(new_loc, curr_means, state.alloc_means, state.phitildes,
                            prior.R, state.u, prior.jump_a, prior.jump_b)
    
    if np.random.uniform() < 0.5:
        # propose birth
        new_loc, curr_locs = birt_prop()
        if np.random.uniform() < birth_arate(new_loc, curr_locs):
            state.non_alloc_means = np.concatenate([curr_locs, [new_loc]])
            state.non_alloc_vars = np.concatenate(
                [state.non_alloc_vars, [tfd.InverseGamma(prior.var_a, prior.var_b).sample()]])
    else:
        del_loc, remaining_locs, idx = death_prop()
        if del_loc is not None:
            a_rate = 1.0 / birth_arate(del_loc, remaining_locs)
            if np.random.uniform() < a_rate:
                state.non_alloc_means = remaining_locs
                state.non_alloc_vars = np.delete(state.non_alloc_vars, idx)

    return state


def update_non_alloc_jumps(state, prior):
    """
    Sample the non allocated jumps from the full conditional distribution
    """
    n_na = state.non_alloc_means.shape[0]
    if n_na > 0:
        state.non_alloc_jumps = np.random.gamma(np.ones(n_na) * prior.jump_a, 
            np.ones(n_na) / (prior.jump_b + state.u))
    else:
        state.non_alloc_jumps = np.array([])
    return state


def update_u(data, state, prior):
    T = np.sum(state.alloc_jumps) + np.sum(state.non_alloc_jumps)
    state.u = np.random.gamma(data.size, 1.0 / T)
    return state


@nb.jit("i4[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])")
def _clus_allocs_update(data, alloc_means, alloc_vars, alloc_jumps, non_alloc_means, 
                        non_alloc_vars, non_alloc_jumps):
    n_a = len(alloc_jumps)
    n_na = len(non_alloc_jumps)
    logprobas = np.empty((len(data), n_a + n_na))
    logprobas[:, :n_a] = uninorm_lpdf_many2many(data, alloc_means, alloc_vars)
    logprobas[:, :n_a] += np.log(alloc_jumps)

    logprobas[:, n_a:] = uninorm_lpdf_many2many(data, non_alloc_means, non_alloc_vars)
    logprobas[:, n_a:] += np.log(non_alloc_jumps)

    out = np.empty(len(data), dtype=np.int32)
    m = np.arange(n_a + n_na)
    for i in range(len(data)):
        out[i] = rand_choice_nb(m, softmax(logprobas[i, :]))
    return out

def clus_allocs_update(data, state):
    state.clus = _clus_allocs_update(data, state.alloc_means, state.alloc_vars, state.alloc_jumps,
                                     state.non_alloc_means, state.non_alloc_vars, state.non_alloc_jumps)
    return state


def relabel(state):

    def vec_translate(a, d):
        return np.vectorize(d.__getitem__)(a)

    active_idx = np.unique(state.clus)
    old2new = {x: i for i, x in enumerate(active_idx)}

    all_means = np.concatenate([state.alloc_means, state.non_alloc_means])
    all_vars = np.concatenate([state.alloc_vars, state.non_alloc_vars])
    all_jumps = np.concatenate([state.alloc_jumps, state.non_alloc_jumps])
    
    state.alloc_means = all_means[active_idx]
    state.non_alloc_means = np.delete(all_means, active_idx)
    state.alloc_vars = all_vars[active_idx]
    state.non_alloc_vars = np.delete(all_vars, active_idx)
    state.alloc_jumps = all_jumps[active_idx]
    state.non_alloc_jumps = np.delete(all_jumps, active_idx)

    state.clus = vec_translate(state.clus, old2new)
    return state

def update_rho(state, prior):
    curr_rho = state.rho
    log_rho = np.log(state.rho)

    # backup stuff
    curr_phis = state.phis
    curr_phitildes = state.phitildes
    curr_dens = dpp_density(state, prior, propto=False)
    curr_prior = gamma_lpdf(curr_rho, prior.rho_a, prior.rho_b) + log_rho

    log_prop_rho = log_rho + np.random.normal(0, 0.25)
    state.rho = np.exp(log_prop_rho)
    state = compute_phis(state)
    prop_dens = dpp_density(state, prior, propto=False)
    prop_prior = gamma_lpdf(state.rho, prior.rho_a, prior.rho_b) + log_prop_rho
    

    log_arate = (prop_dens + prop_prior ) - (
        curr_dens + curr_prior)
    
    assert(isfinite(curr_dens))
    assert(isfinite(prop_dens))
    assert(isfinite(prop_prior))
    assert(isfinite(curr_prior))

    if log_arate < np.log(np.random.uniform()):
        # reject and revert
        state.rho = curr_rho
        state.phis = curr_phis
        state.phitildes = curr_phitildes

    return state

def update_nu(state, prior):
    curr_nu = state.nu
    log_nu = np.log(curr_nu)

    

    # backup stuff
    curr_phis = state.phis
    curr_phitildes = state.phitildes
    curr_dens = dpp_density(state, prior, propto=False)
    curr_prior = gamma_lpdf(curr_nu, prior.nu_a, prior.nu_b) + log_nu

    log_prop_nu = log_nu + np.random.normal(0, 0.25)
    state.nu = np.exp(log_prop_nu)
    state = compute_phis(state)
    prop_dens = dpp_density(state, prior, propto=False)
    prop_prior = gamma_lpdf(state.nu, prior.nu_a, prior.nu_b) + log_prop_nu

    assert(isfinite(curr_dens))
    assert(isfinite(prop_dens))
    assert(isfinite(prop_prior))
    assert(isfinite(curr_prior))

    log_arate = (prop_dens + prop_prior ) - (
        curr_dens + curr_prior)

    if log_arate < np.log(np.random.uniform()):
        # reject and revert
        state.nu = curr_nu
        state.phis = curr_phis
        state.phitildes = curr_phitildes

    return state  

def update_s(state, prior):
    curr_s = state.s
    logit_s = logit(curr_s)

    # backup stuff
    curr_phis = state.phis
    curr_phitildes = state.phitildes
    curr_dens = dpp_density(state, prior, propto=False)
    curr_prior = beta_lpdf(curr_s, prior.s_a, prior.s_b) + \
        np.log(curr_s * (1 - curr_s))

    prop_s = expit(logit_s + np.random.normal(0.0, 0.5))
    state.s = prop_s
    state = compute_phis(state)
    prop_dens = dpp_density(state, prior, propto=False)
    prop_prior = beta_lpdf(prop_s, prior.s_a, prior.s_b) + \
        np.log(prop_s * (1 - prop_s))
    

    log_arate = (prop_dens + prop_prior ) - (
        curr_dens + curr_prior)

    if log_arate < np.log(np.random.uniform()):
        # reject and revert
        state.s = curr_s
        state.phis = curr_phis
        state.phitildes = curr_phitildes

    return state



def step(data, prev_state: State, prior:Prior):
    state = deepcopy(prev_state)
    state.iter = prev_state.iter + 1

    state = update_alloc_atoms(data, state, prior)
    state = update_alloc_jumps(state, prior)
    state = update_non_alloc_atoms(state, prior)    
    state = update_non_alloc_jumps(state, prior)    
    state = clus_allocs_update(data, state)
    state = relabel(state)

    state = update_u(data, state, prior)

    if prior.update_rho:
        state = update_rho(state, prior)
    
    if prior.update_nu:
        state = update_nu(state, prior)

    if prior.update_s:
        state = update_s(state, prior)

    return state
