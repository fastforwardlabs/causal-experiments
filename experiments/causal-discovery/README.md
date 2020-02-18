# causal discovery

Causal discovery is the process of inferring the causal graph structure from observational data. This experiment plays with the causal discovery toolbox.

Install dependencies into python 3.7.x environment with `pip install -r requirements.txt`.
Run notebooks from the notebook directory (for relative directories in code) with `jupyter notebook` or `jupyter lab` or whatever you wish.

To use the causal discovery toolbox, one must additionally install R, and the packages listed in `r-requirements.txt`.
The easiest way to do this is to [install bioconductor](https://www.bioconductor.org/install/), then, in an R REPL, run

```R
BiocManager::install(c("pcalg","kpcalg","bnlearn","sparsebn","SID","CAM","D2C","RCIT"))
```

It turns out not all packages are still available, and CausalDiscoveryToolbox is out of date in this regard (so some functions are broken).

## notebooks

1. Working through the [CausalDiscoveryToolbox](https://diviyan-kalainathan.github.io/CausalDiscoveryToolbox/html/index.html) tutorial and trying some structure learning methods.
