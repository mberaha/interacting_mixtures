import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class State:
    iter: int
    clus: np.array
    alloc_means: np.array
    non_alloc_means: np.array
    alloc_vars: np.array
    non_alloc_vars: np.array
    alloc_jumps: np.array
    non_alloc_jumps: np.array
    u: float
    rho: float
    nu: float
    s: float
    alpha: Optional[float] = None
    phis: Optional[np.array] = None
    phitildes: Optional[np.array] = None


@dataclass
class Prior:
    R: np.array
    mh_sigma: float
    jump_a: float
    jump_b: float
    var_a: float
    var_b: float
    rho: Optional[float] = None 
    nu: Optional[float] = None 
    s: Optional[float] = None
    update_rho: bool = False
    update_s: bool = False
    update_nu: bool = False
    rho_a: Optional[float] = None
    rho_b: Optional[float] = None 
    s_a: Optional[float] = None 
    s_b: Optional[float] = None
    nu_a: Optional[float] = None 
    nu_b: Optional[float] = None
