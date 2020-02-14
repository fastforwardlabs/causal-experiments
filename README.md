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
```

## experiments

* invariant risk minimization - experiments around [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)

## todo

"Doing" an experiment in this case means making the code Chris wrote runnable (so, providing reqs file, adding make target for data etc.)

- [x] invariant risk minimization experiments
- [ ] dowhy experiments
- [ ] causal discovery toolbox experiments
- [ ] invariant causal prediciton experiments
- [ ] move deconfounder repo to an experiment here

## license

This is a repo for playing with things, understanding libraries and experimenting.
The code is currently **unlicensed**, and we accept no liability related to its use.