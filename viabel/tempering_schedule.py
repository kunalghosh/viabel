
import numpy as np

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist





def fixed_tempering(n_iters=10):
    epsilon_list = np.linspace(0., 1., n_iters)
    return epsilon_list



def slow_tempering(n_iters =10, change_point=0.25):
    if n_iters //2 == 1:
        n_iters = n_iters +1

    n_half = n_iters //2

    epsilon_first = np.linspace(0, change_point, n_half)
    epsilon_second = np.linspace(change_point, 1, n_half)
    epsilon_list = np.concatenate((epsilon_first, epsilon_second), axis=0)
    return epsilon_list




