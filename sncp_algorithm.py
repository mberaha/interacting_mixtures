import numpy as np
from copy import deepcopy
from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions


def nnig_update(data, mu0, lam, a, b):
    """
    Samples from the full conditional of the model
        y_1, ..., y_n | mu, sig2 ~ N(mu, sig2)
                       mu | sig2 ~ N(mu0, sig2/ lam)
                            sig2 ~ IG(a, b)
    """
    ybar = np.mean(data)
    var = np.var(data)
    card = len(data)

    post_mean = (lam * mu0 + ybar * card) / (lam + card)
    post_lam = lam + card
    post_shape = a + 0.5 * card
    post_rate = b + \
        0.5 * var * card + \
        0.5 * lam * card * (mu0 - ybar)**2 / post_lam
    var_post = tfd.InverseGamma(post_shape, post_rate).sample()
    mu_post = tfd.Normal(post_mean, np.sqrt(var_post /post_lam)).sample()
    return np.array([mu_post, var_post])


def clus_allocs_update(data, state):
    n_a = len(state.alloc_jumps)
    n_na = len(state.non_alloc_jumps)
    probas = np.zeros((len(data), n_a + n_na))
    probas[:, :n_a] = tfd.Normal(
        state.alloc_atoms[:, 0], 
        np.sqrt(state.alloc_atoms[:, 1])).log_prob(data[:, np.newaxis])
    probas[:, :n_a] += np.log(state.alloc_jumps)
    
    probas[:, n_a:] = tfd.Normal(
        state.non_alloc_atoms[:, 0], 
        np.sqrt(state.non_alloc_atoms[:, 1])).log_prob(data[:, np.newaxis])+ \
            np.log(state.non_alloc_jumps)

    state.clus = tfd.Categorical(logits=probas).sample().T
    return state


def relabel(state):

    def vec_translate(a, d):
        return np.vectorize(d.__getitem__)(a)

    active_idx = np.unique(state.clus)
    old2new = {x: i for i, x in enumerate(active_idx)}
    all_atoms = np.vstack([state.alloc_atoms, state.non_alloc_atoms])
    all_jumps = np.concatenate([state.alloc_jumps, state.non_alloc_jumps])
    
    state.alloc_atoms = all_atoms[active_idx, :]
    state.non_alloc_atoms = np.delete(all_atoms, active_idx, 0)
    state.alloc_jumps = all_jumps[active_idx]
    state.non_alloc_jumps = np.delete(all_jumps, active_idx)

    state.clus = vec_translate(state.clus, old2new)
    state.active_t_vals = state.t_vals[active_idx]
    return state


def update_alloc_jumps(state, prior):
    cluscount = np.sum(state.clus[:, np.newaxis] == 
                       np.arange(state.alloc_jumps.shape[0]), axis=0)
    state.alloc_jumps = tfd.Gamma(
        cluscount + prior.jump_a, prior.jump_b + state.u).sample()
    return state


def update_non_alloc_jumps(state, prior):
    n_na = state.non_alloc_atoms.shape[0]
    state.non_alloc_jumps = tfd.Gamma(np.ones(n_na) * prior.jump_a, 
        np.ones(n_na) * (prior.jump_b + state.u)).sample()
    return state


def update_alloc_atoms(data, state, prior):
    data_by_clus = [data[state.clus == i] for i in np.unique(state.clus)]
    stats = np.array([(len(x), np.mean(x), np.var(x)) for x in data_by_clus])
    k = len(data_by_clus)

    post_vars = 1.0 / (stats[:, 0] / state.alloc_atoms[:k, 1] + 1.0 / prior.alpha**2)
    post_means = (state.latent_centers[state.t_vals[:k]] / prior.alpha ** 2 + 
                stats[:, 0] * stats[:, 1] / state.alloc_atoms[:k, 1]) * post_vars
    state.alloc_atoms[:, 0] = tfd.Normal(post_means, np.sqrt(post_vars)).sample()

    post_a = prior.var_a + stats[:, 0] / 2
    post_b = prior.var_b + 0.5 * stats[:, 2] * stats[:, 0]
    state.alloc_atoms[:, 1] = tfd.InverseGamma(post_a, post_b).sample()
    return state


def update_non_alloc_atoms(state, prior):
    def sample_many_atoms(means, std, n_points, base_t):
        out = []
        t_vals = []
        for i, c in enumerate(means):
            num_p = int(n_points[i])
            means = tfd.Normal(c, std).sample(num_p)
            vars = tfd.InverseGamma(prior.var_a, prior.var_b).sample(num_p)
            # print("means: ", means)
            # print("vars: ", vars)
            # print("xxxxxxx")
            curr = np.hstack([means.reshape(-1, 1), vars.reshape(-1, 1)])
            out.append(curr)
            t_vals.append(np.ones(n_points[i]) * (i + base_t))
        t_vals = np.concatenate(t_vals)
        return out, t_vals

    # zeta_vars = 1.0 / (1 / prior.big_var + 1 / prior.alpha ** 2)
    # zeta_means = (state.alloc_atoms[:, 0] / (prior.alpha**2) + 
    #               prior.big_mean / (prior.big_var)) * zeta_vars
    # zetas = tfd.Normal(zeta_means, np.sqrt(zeta_vars)).sample()
    zetas = state.latent_centers[np.unique(state.active_t_vals)].astype(float)

    poi_intensity = prior.gamma * np.exp(-prior.gamma * (
        1 - (prior.jump_b / (prior.jump_b + state.u))**prior.jump_a))
    n_points = tfd.Poisson(poi_intensity).sample(len(zetas)).astype(int)
    out, _ = sample_many_atoms(zetas, prior.alpha, n_points, 0)
    t_vals = []
    for i, n in enumerate(n_points):
        t_vals.append(np.ones(n) * state.t_vals[i])
    t_vals = np.concatenate(t_vals)

    # simulate the shot noise cox process 
    sncp_points =  tfd.Poisson(poi_intensity).sample()
    new_centers = []
    if (sncp_points > 0):
        new_centers = tfd.Normal(prior.big_mean, prior.big_var).sample(sncp_points)
        n_points = tfd.Poisson(prior.gamma).sample(sncp_points).astype(int)
        new_centers = new_centers[n_points > 0]
        n_points = n_points[n_points > 0]
        
        if len(n_points):
            out_temp, t_temp = sample_many_atoms(
                new_centers, prior.alpha, n_points, np.max(state.t_vals) + 1)
            out.extend(out_temp)
            t_vals = np.concatenate([t_vals, t_temp])
    if len(out):
        state.non_alloc_atoms = np.vstack(out)
    state.non_active_t_vals = t_vals
    return state, new_centers


def update_latent_centers(data, state, prior):
    # all_centers = state.alloc_atoms[:, 0]
    all_centers = np.concatenate([
        state.alloc_atoms[:, 0], state.non_alloc_atoms[:, 0]])
    t_vals = state.t_vals[:all_centers.shape[0]]
    data_by_clus = [data[state.t_vals[state.clus] == i] for i in np.unique(t_vals)]
    stats = np.array([(len(x), np.mean(x), np.var(x)) for x in data_by_clus])
    post_vars = 1.0 / (1.0 / prior.big_var + stats[:, 0] / prior.alpha**2) 
    post_means = post_vars * (
        prior.big_mean / prior.big_var + stats[:, 0] * stats[:, 1] / prior.alpha**2)
    post_vars[stats[:, 0] == 0] = prior.big_var
    post_means[stats[:, 0] == 0] = prior.big_mean
    state.latent_centers = tfd.Normal(post_means, np.sqrt(post_vars)).sample()
    return state


def relabel_t_vals(state):
    def vec_translate(a, d):
        return np.vectorize(d.__getitem__)(a)

    t_vals = state.t_vals
    active_idx = np.unique(t_vals)
    old2new = {x: i for i, x in enumerate(active_idx)}
    state.latent_centers = state.latent_centers[active_idx]
    state.t_vals = vec_translate(t_vals, old2new)
    return state

def update_t_vals(state, prior): 
    all_centers = np.concatenate([state.alloc_atoms[:, 0], state.non_alloc_atoms[:, 0]])
    probas = tfd.Normal(state.latent_centers, prior.alpha).log_prob(all_centers[:, np.newaxis])
    state.t_vals = tfd.Categorical(logits=probas).sample()
    return relabel_t_vals(state)

def update_u(data, state, prior):
    state.u = tfd.Gamma(data.size, 
                       np.sum(state.alloc_jumps) + np.sum(state.non_alloc_jumps)).sample()
    return state


def step(data, prev_state, prior):
    state = deepcopy(prev_state)
    # print("number latent scenters 0: {0}, max_t : {1}".format(
    #     len(state.latent_centers), np.max(state.t_vals)))
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))
    state.iter = prev_state.iter + 1
   
    state = update_alloc_jumps(state, prior)
    # state = update_alloc_atoms(data, state, prior)
    # print("number latent scenters 1: {0}, max_t : {1}".format(
    #     len(state.latent_centers), np.max(state.t_vals)))
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))
    state, non_active_centers = update_non_alloc_atoms(state, prior)    
    # print("non_active_centers: ", non_active_centers)
   
    # state.latent_centers = np.concatenate([
    #     state.latent_centers[np.unique(state.active_t_vals)], non_active_centers])
    state.latent_centers = np.concatenate([
        state.latent_centers, non_active_centers])
    state.t_vals = np.concatenate([state.active_t_vals, 
                                   state.non_active_t_vals]).astype(int)
    state = relabel_t_vals(state)
    # print("number latent scenters 2: {0}, max_t : {1}".format(
    #     len(state.latent_centers), np.max(state.t_vals)))
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))
    
    state = update_non_alloc_jumps(state, prior)    
    
    state = clus_allocs_update(data, state) 
    state = relabel(state)
    # print("number latent scenters 3: {0}, max_t : {1}".format(
    #     len(state.latent_centers), np.max(state.t_vals)))
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))

    state = update_latent_centers(data, state, prior)
    # print("number latent scenters 4: {0}, max_t : {1}".format(
    #     len(state.latent_centers), np.max(state.t_vals)))
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))

    state = update_t_vals(state, prior)
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))

    state = update_u(data, state, prior)
    state = relabel(state)
    assert(len(state.latent_centers) ==  (np.max(state.t_vals) + 1))
    
    # print("*************")
    return state


def get_dens(state, grid):
    eval_comps = tfd.Normal(state.alloc_atoms[:, 0], np.sqrt(state.alloc_atoms[:, 1])).prob(grid[:, np.newaxis]) 
    weights = state.alloc_jumps
    weights /= np.sum(weights)
    dens = np.sum(eval_comps * weights, axis=1)
    return dens
