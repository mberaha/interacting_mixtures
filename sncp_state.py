import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class State:
    iter: int
    clus: np.array
    alloc_atoms: np.array
    non_alloc_atoms: np.array
    alloc_vars: np.array
    alloc_jumps: np.array
    non_alloc_jumps: np.array
    u: float
    latent_centers: np.array
    t_vals: np.array
    active_t_vals: Optional[np.array] = None
    non_active_t_vals: Optional[np.array] = None


@dataclass
class Prior:
    gamma: float
    big_mean: float
    big_var: float
    jump_a: float
    jump_b: float
    var_a: float
    var_b: float
    alpha: float
