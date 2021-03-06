## environment setup

From inside a suitable python 3.7 virtual environment:

```bash
pip install -r requirements.txt
```

## package

The code has currently been adopted from Facebook's code repo - [InvariantRiskMinimization](https://github.com/facebookresearch/InvariantRiskMinimization)

## notebooks

Run the notebooks with `jupyter lab` from a virtual env with the dependencies installed.

1. 1-invariant-risk-minimization-minimal-example.ipynb - Just a replication of the example pytorch code provided in the paper.
2. 2-invariant-risk-minimization-on-news.ipynb - Extending the example code to work on a real dataset. Linear regression fails.
3. 3-overfit-the-news.ipynb - Try overfitting the news popularity dataset, just to assess how learnable a problem this is. Turns out, not very.
4. 4-irm-on-churn-data.ipynb - Trying both IRM and regular ERM (ie. normal supervised learning) on churn dataset. Environments defined by Dependents and Partners seem to be too similar.
5-9. Experiments with IRM on the UCI ML repo bikeshare dataset, trying daily and hourly data, Bayesian Opitimisation for hyperparameter search and scaling data. In all cases, it seems neural nets are too fickle, and a good model cannot be learned with IRM. Without IRM, even the best found models are very initialisation sensitive.