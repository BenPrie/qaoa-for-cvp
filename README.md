# Practical Quantum Algorithms for Cryptanalysis

`requirements.txt` should be sufficient to run any `.py` scripts. There may be additional dependencies within notebooks that should be installed if and when necessary (e.g. tqdm).

## Overview of Notebook Contents

- `Defining the SQIF.ipynb` walks through the SQIF algorithm in [Yan et al. (2022)](https://arxiv.org/abs/2212.12372) step-by-step. It proceeds right the way up to the linear system of equations that one solves to reveal factors, though does not do this because (1) that is not what we care about in this project, and (2) we agree with [Grebnev et al. (2023)](https://arxiv.org/html/2303.04656v6) that there are cases for which not enough sr-pairs can be found to successfully factor (and give the example).
- `Building a Hamiltonian.ipynb` concisely presents the implementation of our code that derives a Hamiltonian for a given semi-prime and hyperparameters. There is no walkthrough -- this is given elsewhere.
- `Sampling by QAOA.ipynb` walks through the process of sampling solutions by measuring the output states of the QAOA circuit.
- `Reproducing Experiments.ipynb` looks at the sampled solutions our implementation obtains for the cases used in [Yan et al. (2022)](https://arxiv.org/abs/2212.12372).
- `5-qubit-QAOA.ipynb` looks in a little more detail at a particular instance (in 5 qubits).
- `VQA for CVP.ipynb` and `QAOA Scaling Analysis.ipynb`  contain all of our data generation and data analysis. Most of our findings and novel contributions appear in this notebook.

Some of this functionality is collated into scripts for ease of use -- those being `sqif.py` and `sqif_algorithm.py`, which offer different flavours of the same thing.