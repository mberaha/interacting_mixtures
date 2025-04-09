# Bayesian Mixture Models with Repulsive and Attractive Atoms

This repository contains the code supporting the paper ``Bayesian Mixture Models with Repulsive and Attractive Atoms'' by Beraha, Argiento, Camerlenghi, and Guglielmi.

The posterior inference algorithms for the SNCP mixture model and DPP mixture model can be found in the files `sncp_algorithm.py` and `dpp_algorithm.py`, respectively.

Moreover

- `Prior Simulation.ipynb` contains code to reproduce Figure 1, i.e., the probability of displaying particular unique values in the sample

- `SNCP Simulation.ipynb` contains code to reproduce the analyses in Section 5.1 and 5.3

- `shapley_analysis.ipynb` contains code to reproduce the analysis in Section 5.3

- To reproduce the analyses in Appendix I, run the scripts `simulations_dpp1.py` and `simulations_dpp2.py`and then execute the notebook `Plots.ipynb`.


The code depends on the following requirements

```
joblib
numpy
matplotlib
tensorflow_probability
scipy
seaborn
```

which can be installed via `pip`. Moreover, you should also install the `bayesmixpy` package as described [here](https://github.com/bayesmix-dev/bayesmix/tree/master/python).
