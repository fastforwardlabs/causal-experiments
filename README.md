# causality experiments

A monorepo for causality-related experiments.

## structure

Data is stored "centrally" in the `data/` folder.
Data is not committed to the repo (some of it is tens of MB and no-one likes a long repo clone), but rather provide Make targets to download it.

Experiments map roughly to "playing with a relevant library" and have their own directories inside the `experiments/` directory.
Notebooks should be written such that the notebook server is run from the relevant subdirectory (for instance, the dowhy notebook server should be run from `causality-experiments/experiments/dowhy`).
Each experiment subdirectory should have its own environment file, and use it's own (python) environment, to avoid library version conflicts.

## getting started

First, clone the repo.
Then, run `make dirs` to make the data directory (which is `.gitignore`d).

To fetch the data, run:
```bash
make news
make churn
make housing
make bikes
make london-bikes
make student
```

(Unfortunately, churn, housing and london-bikes will just give you instructions on where to find the data).

## experiments

* *causal discovery* - playing with the [CausalDiscoveryToolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
* *causalnex* - trying the new [CausalNex](https://causalnex.readthedocs.io/en/latest/) library
* *dowhy* - trying to use [DoWhy](https://microsoft.github.io/dowhy/) on a couple of datasets
* *invariant causal prediction* - toying with the R libraries accompanying the papers [Causal inference using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332) and [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)
* *invariant risk minimization* - experiments around [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)


## todo

"Doing" an experiment in this case means making the code Chris wrote runnable (so, providing reqs file, adding make target for data etc.)

- [x] invariant risk minimization experiments
- [x] dowhy experiments
- [x] causal discovery toolbox experiments
- [x] invariant causal prediciton experiments
- [ ] move deconfounder repo to an experiment here

## license

This is a repo for playing with things, understanding libraries and experimenting.
The code is currently **unlicensed**, and we accept no liability related to its use.
