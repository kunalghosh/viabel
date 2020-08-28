from collections import namedtuple

from autograd import value_and_grad, vector_jacobian_product, jacobian, elementwise_grad
from autograd.extend import primitive, defvjp

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm
from scipy.linalg import eigvalsh
from  .optimization_diagnostics import autocorrelation, monte_carlo_se, monte_carlo_se2, compute_khat_iterates, gpdfit
from .psis import psislw

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput)

from functools import partial

import tqdm
import scipy.stats as stats
from .tempering_schedule import  adaptive_tempering, sigmoid_tempering, telescope_tempering

from ._distributions import multivariate_t_logpdf

from .functions import compute_R_hat, compute_R_hat_adaptive_numpy, compute_R_hat_halfway, stochastic_iterate_averaging
from .functions import flat_to_triang, triang_to_flat
from .vb import mean_field_gaussian_variational_family, make_stan_log_density,  \
    learning_rate_schedule




def black_box_gapis(var_family, logdensity, logdensity_grad, n_samples, k):
    def compute_log_weights(var_param, epsilon, seed):
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples)*epsilon - var_family.logdensity(samples, var_param)


    def compute_Hessian(var_param, delta=1e-4):
        hessian = np.zeros((k,k))
        obj_mean_grad1 = logdensity_grad(var_param[:k] + delta)
        obj_mean_grad2 = logdensity_grad(var_param[:k] -delta)
        for i in range(k):
            hessian[i,i] = (obj_mean_grad1[i] -obj_mean_grad2[i])/2*delta
        return hessian

    def objective_grad_and_log_norm(var_param):
        var_param[:k] = var_param[:k]
        obj_grad= logdensity_grad(var_param[:k])
        hessian = compute_Hessian(var_param)
        return 0., obj_grad, hessian

    return objective_grad_and_log_norm



def adagrad_optimize_IS(k, n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, learning_rate_end=None, ):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    pareto_k_list = []
    neff_list = []
    prev_z = np.zeros((1,k))

    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                obj_val, obj_grad, hessian = objective_and_grad(variational_param)
                value_history.append(obj_val)
                local_grad_history.append(obj_grad)
                #grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                #scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                #accum_sum = np.sum(scaled_grads ** 2, axis=0)
                variational_param = variational_param + curr_learning_rate * obj_grad
                if i >= 3 * n_iters // 4:
                    variational_param_history.append(variational_param.copy())
                if i % 10 == 0:
                    avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            progress.close()
        finally:
            progress.close()

    return variational_param, hessian







