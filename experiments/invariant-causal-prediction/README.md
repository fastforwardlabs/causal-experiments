# Invariant causal prediction

Experimenting with the R libraries `InvariantCausalPrediction` and `nonlinearICP`, associated, respectively, with the papers:
* [Causal inference using invariant prediction: identification and confidence intervals
](https://arxiv.org/abs/1501.01332)
* [Invariant Causal Prediction for Nonlinear Models
](https://arxiv.org/abs/1706.08576)

Some R dependencies are necessary.
I'm not aware of R having any sort of dependency management outside of writing a package, but the key libraries may be installed from the R console with:

```R
install.packages(c("tidyverse","InvariantCausalPrediction", "nonlinearICP", "CondIndTests"))
```

The (PDF) docs for the three stats packages are here:

* [InvariantCausalPrediction](https://cran.r-project.org/web/packages/InvariantCausalPrediction/InvariantCausalPrediction.pdf)
* [CondIndTests](https://cran.r-project.org/web/packages/CondIndTests/CondIndTests.pdf)
* [nonlinearICP](https://cran.r-project.org/web/packages/nonlinearICP/nonlinearICP.pdf)

## scripts
1. 1-linear-icp-churn.R - Trying InvariantCausalPrediction on the churn dataset, with Partner, Dependents and SeniorCitizen defining environments.
2. 2-nonlinear-icp-housing.R - nonlinearICP on california housing dataset, with lat and long defining environments.
3. 3-nonlinear-icp-bikesharing.R - nonlinearICP on bike sharing dataset. Environment split across time. Model rejected.
4. 4-linear-icp-churn-again.R - As first script, with more notes explaining what's going on. 
5. 5-linear-news-icp.R - ICP on news popularity dataset. Linear model rejected.
