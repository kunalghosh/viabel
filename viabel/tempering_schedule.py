
import numpy as np

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist


def fixed_tempering(n_iters=10):
    if n_iters==1:
        return np.array([0., 1.])
    beta_list = np.linspace(0., 1., n_iters)
    return beta_list


def inverse_tempering1(n_iters=1000):
    betas= []
    for i in range(0, n_iters):
        betas.append(2**i)

    betas= np.array(betas)
    betas = 1. /betas
    betas = 1. - betas
    return betas


def sigmoid_tempering(n_iters=1000, limit=5):
    if n_iters==1:
        return np.array([0., 1.])

    a = np.linspace(-5, 5, n_iters)
    betas = 1. /(1. + np.exp(-a))
    return betas



def telescope_tempering(n_iters= 1000):
    betas = [0]
    for i in range(n_iters, 0, -1):
        betas.append(1./2**n_iters)

    return np.array(betas)


def logarithmic_tempering(n_iters=1000, normalise=False):
    logbeta_list = np.linspace(0., 1., n_iters)
    beta_list = np.exp(logbeta_list)
    if normalise:
        beta_list = beta_list/np.sum(beta_list)
    return beta_list


def slow_tempering(n_iters =10, change_point=0.25):
    if n_iters //2 == 1:
        n_iters = n_iters +1

    n_half = n_iters //2

    epsilon_first = np.linspace(0, change_point, n_half)
    epsilon_second = np.linspace(change_point, 1, n_half)
    epsilon_list = np.concatenate((epsilon_first, epsilon_second), axis=0)
    return epsilon_list


def adaptive_tempering(beta_current, Neff_prev, Neff_current, delta=0.01):
    #adapt beta according to beta_current!!!
    beta_prev = beta_current
    if Neff_current > Neff_prev:
        beta_current = beta_current + delta
    return beta_current





