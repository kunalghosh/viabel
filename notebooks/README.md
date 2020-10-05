###### README file ######

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
 