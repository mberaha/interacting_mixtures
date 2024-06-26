import numpy as np
import pandas as pd
import pickle

from scipy.stats import t
from scipy.integrate import trapz
from copy import deepcopy
from itertools import product
from joblib import Parallel, delayed

import dpp_algorithm as dpp
from dpp_state import State, Prior
from utils.common import *
from utils.mfm import run_mfm


NDATA = [500]

N_ITER = 5000
N_BURN = 25000

default_prior = Prior(
    R=np.array([-15.0, 15.0]),
    mh_sigma=2.0,
    jump_a=2.0,
    jump_b=1.0,
    var_a=3.0,
    var_b=3.0,
    nu = 2.0,
    s = 0.75,
    rho = 2.0,
)


def eval_true_dens(x):
    return 0.5 * t.pdf(x, 3, loc=-5) + 0.5 * t.pdf(x, 3, loc=5)


def hellinger(f, g, xgrid, squared=True):
    out = 0.5 * trapz((np.sqrt(f) - np.sqrt(g))**2, xgrid)
    if not squared:
        out = np.sqrt(out)
    return out


def tv_dist(f, g, xgrid):
    out = 0.5 * trapz(np.abs(f - g), xgrid)
    return out


def generate_data(ndata):
    mix_component = np.random.choice(2, size=ndata)
    means = np.array([-5, 5])
    data = t.rvs(3, loc=means[mix_component])
    return data, mix_component


def initialize_state(data, prior):
    M = 3
    state = State(
        iter=0,
        clus = np.random.choice(np.arange(M), size=len(data)),
        alloc_means = np.random.uniform(prior.R[0], prior.R[1], size=M),
        non_alloc_means = np.random.uniform(prior.R[0], prior.R[1], size=M),
        alloc_vars = np.ones(M),
        non_alloc_vars=np.ones(M),
        alloc_jumps=np.ones(M),
        non_alloc_jumps=np.ones(M),
        u=len(data) / 20.0,
        rho = prior.rho,
        nu = prior.nu,
        s = prior.s)
    
    state = dpp.compute_phis(state)
    state = dpp.clus_allocs_update(data, state)
    state = dpp.relabel(state)
    return state


def run_mcmc(data, state, prior):
    for _ in range(N_BURN):
        state = dpp.step(data, state, prior)

    states = [state]
    for _ in range(N_ITER):
        state = dpp.step(data, state, prior)
        saved_state = deepcopy(state)
        saved_state.phis = np.array([])
        saved_state.phitildes = np.array([])
        states.append(saved_state)
    return states


def estimate_dens(states, xgrid):
    out = np.zeros_like(xgrid)
    for s in states:
        means = np.concatenate([s.alloc_means, s.non_alloc_means])
        vars = np.concatenate([s.alloc_vars, s.non_alloc_vars])
        weights = np.concatenate([s.alloc_jumps, s.non_alloc_jumps])
        out += eval_mixture_density(xgrid, means, vars, weights)

    out /= len(states)
    return out


def stats_from_chains(chain, true_dens, xgrid, iternum, prior):
    estim_dens = estimate_dens(chain, xgrid)
    hell = hellinger(true_dens, estim_dens, xgrid)
    tv = tv_dist(true_dens, estim_dens, xgrid)
    avg_nclus = np.mean([
        len(np.unique(s.clus)) for s in chain
    ])

    return {
        "hell": hell,
        "tv": tv_dist,
        "nlcus": avg_nclus,
        "iter": iternum,
        "update_rho": prior.update_rho,
        "update_s": prior.update_s,
        "update_nu": prior.update_nu
    }


def run_simulation(iternum):
    xgrid = np.linspace(-15, 15, 5000)
    true_dens = eval_true_dens(xgrid)
    stats = []
    data, true_clus = generate_data(NDATA)

    # update only s
    prior = deepcopy(default_prior)
    prior.update_s = True
    prior.s_a = 2.0
    prior.s_b = 2.0

    state = initialize_state(data, prior)
    chain = run_mcmc(data, state, prior)
    curr_stats = stats_from_chains(chain, true_dens, xgrid, iternum, prior)
    stats.append(curr_stats)
    if iternum == 0:
        with open("chains_s.pickle", "wb") as fp:
            pickle.dump(chain, fp)

    # update only nu
    prior = deepcopy(default_prior)
    prior.update_nu = True
    prior.nu_a = 4.0
    prior.nu_b = 2.0

    state = initialize_state(data, prior)
    chain = run_mcmc(data, state, prior)
    curr_stats = stats_from_chains(chain, true_dens, xgrid, iternum, prior)
    stats.append(curr_stats)
    if iternum == 0:
        with open("chains_nu.pickle", "wb") as fp:
            pickle.dump(chain, fp)


    # update rho and s
    prior = deepcopy(default_prior)
    prior.update_rho = True
    prior.rho_a = 4.0
    prior.rho_b = 2.0
    prior.update_s = True
    prior.s_a = 2.0
    prior.s_b = 2.0
    prior.update_nu = False

    state = initialize_state(data, prior)
    chain = run_mcmc(data, state, prior)
    curr_stats = stats_from_chains(chain, true_dens, xgrid, iternum, prior)
    stats.append(curr_stats)

    # update all
    prior = deepcopy(default_prior)
    prior.update_rho = True
    prior.rho_a = 4.0
    prior.rho_b = 2.0
    prior.update_nu = True
    prior.nu_a = 4.0
    prior.nu_b = 2.0
    prior.update_s = True
    prior.s_a = 2.0
    prior.s_b = 2.0

    state = initialize_state(data, prior)
    chain = run_mcmc(data, state, prior)
    curr_stats = stats_from_chains(chain, true_dens, xgrid, iternum, prior)
    curr_stats["update_rho"] = True
    curr_stats["update_s"] = False
    curr_stats["update_nu"] = True
    stats.append(curr_stats)

    return pd.DataFrame(stats)

if __name__ == "__main__":
    NJOBS = 20
    NREP = 100
    dfs = Parallel(n_jobs=NJOBS)(
        delayed(run_simulation)(i) for i in range(NREP))
    out = pd.concat(dfs)
    pd.to_pickle(out, "dpp2_simulation_out.pickle")

    # run_simulation(0)
