import numpy as np
from bayesmixpy import build_bayesmix, run_mcmc

mfm_params = """
fixed_value {
    lambda: 4.0
    gamma: 1.0
}
"""

dp_params = """
fixed_value {
    totalmass: 2.0
}
"""

g0_params = """
fixed_values {
    mean: 0.0
    var_scaling: 0.05
    shape: 2.0
    scale: 2.0
}
"""

algo_params = """
    algo_id: "Neal2"
    rng_seed: 20201124
    iterations: 10000
    burnin: 5000
    init_num_clusters: 3
    neal8_n_aux: 10
"""


def run_mfm(data, xgrid):
    log_dens, nclus, _, _ , _= run_mcmc(
        "NNIG", "DP", data, g0_params, dp_params, algo_params, 
        dens_grid=xgrid, return_clusters=False, return_num_clusters=True,
        return_best_clus=False)
    
    estim_dens = np.mean(np.exp(log_dens), axis=0)
    avg_nclus = np.median(nclus)
    return estim_dens, avg_nclus