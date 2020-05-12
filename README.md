#  `viabel`: Variational Inference Approximation Bounds that are Efficient and Lightweight



## Structure

The vb.py contains all the basic vb functions, making objective
functions based on divergence measure and the gradient 
function like black_box_klvi, black_box_chivi, black_box_klvi_pd(with
path gradients) etc. 
 The next set of functions it contains all the approximate
 families: mean field gaussian, t- distribution.  The full
 rank Gaussian is obtained using t-distribution and passing a very high
 degree of freedom. 
 The optimizer.py file contains all the optimisers used, currently
 it supports three: Adagrad, RMSProp, and ADAM. 
 
 The file optimisation_diagnostics contains functions for
 diagnosing convergence such as: Rhat function, k-hat function , autocorrelation
 function(acf) and monte carlo standard error (mcse) which all take
 chains of iterates and use all the chains and iterates to compute
 those statistics for each variational parameter of the model.
 
 The file optimizer_stopping_rule.py file contains the same optimisers
 with some modifications made to be used as automatic stopping using
 some stopping rule. These files will be merged soon.
 
 The file tempering_Schedule contains functions for scheduling 
 tempering. 
 The file functions.py contains some more utility functions.


## Description

This package computes bounds errors of the mean, standard deviation, and variances
estimates produced by a continuous approximation to a (unnormalized) distribution.
A canonical application is a variational approximation to a Bayesian posterior
distribution.
In particular, using samples from the approximation *Q* and evaluations of the
(maybe unnormalized) log densities of *Q* and (target distribution) *P*,
the package provides functionality to compute bounds on:

* the &alpha;-divergence between *P* and *Q*
* the *p*-Wasserstein distance between *P* and *Q*
* the differences between the means, standard deviations, and variances of *P* and *Q*

There is also an (optional) variational Bayes functionality (`viabel.vb`), which
supports both standard KL-based variational inference (KLVI) and chi-squared
variational inference (CHIVI).
Models are provided as `autograd`-compatible log densities or can be constructed
from `pystan` fit objects.
The variational objective is optimized using a windowed version of adagrad
and unbiased reparameterization gradients.
By default there is support for mean-field Gaussian, mean-field Student's t,
and full-rank Student's t variational families.

If you use this package, please cite:

[Validated Variational Inference via Practical Posterior Error Bounds](https://arxiv.org/abs/1910.04102).
Jonathan H. Huggins,
Miko&#0322;aj Kasprzak,
Trevor Campbell,
Tamara Broderick.
In *Proc. of the 23rd International Conference on Artificial Intelligence and
Statistics* (AISTATS), Palermo, Italy. PMLR: Volume 108, 2020.

## How to install

If you just want to compute bounds, just run `pip install viabel`.
The only dependency is `numpy`.
If you want to use the basic variational Bayes
functionality, run `pip install viabel[vb]`.
This will install some additional dependencies.
If in addition to the above, you want to run all of the [example notebooks](notebooks),
use the command `pip install viabel[vb]`, which will install even more dependencies.

## Usage Examples

The [normal mixture notebook](notebooks/normal-mixture.ipynb) provides basic
usage examples of the bounds.

The [robust regression example](notebooks/robust-regression.ipynb) demonstrates
how to use the variational Bayes functionality and then compute bounds.

## Running Comparison Experiments

The [notebooks/experiments.py](notebooks/experiments.py) contains additional
functionality for running experiments and computing PSIS-corrected posterior estimates.
The [robust regression example](notebooks/robust-regression.ipynb) uses some of this functionality.
A simple [funnel distribution example](notebooks/funnel-distribution.ipynb) demonstrates how to use the high-level `run_experiment` function.
The [eight schools example](notebooks/eight-schools.ipynb) is more involved and realistic.

## Development and testing

After cloning the git repository, run `nosetests` to test the package.
Currently there is only coverage for `viabel.bounds`.
