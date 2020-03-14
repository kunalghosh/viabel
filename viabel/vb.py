from collections import namedtuple

from autograd import value_and_grad, vector_jacobian_product
from autograd.extend import primitive, defvjp

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm
from scipy.linalg import eigvalsh

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput)

from functools import partial

import tqdm
import scipy.stats as stats

from ._distributions import multivariate_t_logpdf

from .functions import compute_R_hat, compute_R_hat_adaptive_numpy, compute_R_hat_halfway, stochastic_iterate_averaging
from .functions import flat_to_triang, triang_to_flat

__all__ = [
    'mean_field_gaussian_variational_family',
    'mean_field_t_variational_family',
    't_variational_family',
    'black_box_klvi',
    'black_box_klvi_pd',
    'black_box_klvi_pd2',
    'black_box_chivi',
    'make_stan_log_density',
    'adagrad_optimize',
    'rmsprop_IA_optimize_with_rhat',
    'adam_IA_optimize_with_rhat'
]

VariationalFamily = namedtuple('VariationalFamily',
                               ['sample', 'entropy',
                                'logdensity', 'mean_and_cov',
                                'pth_moment', 'var_param_dim'])


def mean_field_gaussian_variational_family(dim):
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_std = var_param[:dim], var_param[dim:]
        return mean, log_std

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_std = unpack_params(var_param)
        return my_rs.randn(n_samples, dim) * np.exp(log_std) + mean

    def entropy(var_param):
        mean, log_std = unpack_params(var_param)
        return 0.5 * dim * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def logdensity(x, var_param):
        mean, log_std = unpack_params(var_param)
        return mvn.logpdf(x, mean, np.diag(np.exp(2*log_std)))

    def mean_and_cov(var_param):
        mean, log_std = unpack_params(var_param)
        return mean, np.diag(np.exp(2*log_std))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        _, log_std = unpack_params(var_param)
        vars = np.exp(2*log_std)
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2*np.sum(vars**2) + np.sum(vars)**2

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)


def full_rank_gaussian_variational_family(dim):
    rs = npr.RandomState(0)

    def beta_to_L(beta):
        print(beta.shape)
        L = flat_to_triang(beta)
        L= L[0]
        return L

    def L_to_beta(L):
        return triang_to_flat(L)

    def unpack_params(var_param):
        mean, beta = var_param[:dim], var_param[dim:]
        return mean, beta

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta)
        return np.dot( my_rs.randn(n_samples, dim), L) + mean
        #return my_rs.randn(n_samples, dim) @ L + mean

    def entropy(var_param):
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta[:,np.newaxis])
        return np.sum(np.log(np.diag(L))) + 0.5*dim* (1 + np.log(2 * np.pi))
        #return 0.5 * dim * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def logdensity(x, var_param):
        mean, beta = unpack_params(var_param)
        L = beta_to_L(beta[:,np.newaxis])
        Sigma = L@L.T
        return mvn.logpdf(x, mean, Sigma)

    def mean_and_cov(var_param):
        mean, beta = unpack_params(var_param)
        L =  beta_to_L(beta[:,np.newaxis])
        Sigma = L@L.T
        return mean, np.diag(Sigma)

    return VariationalFamily(sample, entropy, logdensity, mean_and_cov, dim*(dim+3)//2)

def mean_field_t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_scale = var_param[:dim], var_param[dim:]
        return mean, log_scale

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_scale = unpack_params(var_param)
        return mean + np.exp(log_scale)*my_rs.standard_t(df, size=(n_samples, dim))

    def entropy(var_param):
        # ignore terms that depend only on df
        mean, log_scale = unpack_params(var_param)
        return np.sum(log_scale)

    def logdensity(x, var_param):
        mean, log_scale = unpack_params(var_param)
        if x.ndim == 1:
            x = x[np.newaxis,:]
        return np.sum(t_dist.logpdf(x, df, mean, np.exp(log_scale)), axis=-1)

    def mean_and_cov(var_param):
        mean, log_scale = unpack_params(var_param)
        return mean, df / (df - 2) * np.diag(np.exp(2*log_scale))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        _, log_scale = unpack_params(var_param)
        scales = np.exp(log_scale)
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(scales**2)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(scales**4) + np.sum(scales**2)**2)

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)


def _get_mu_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['Sigma'] = PSDSymmetricMatrixPattern(size=dim)
    return ms_pattern


def t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    ms_pattern = _get_mu_sigma_pattern(dim)

    logdensity = FlattenFunctionInput(
        lambda x, ms_dict: multivariate_t_logpdf(x, ms_dict['mu'], ms_dict['Sigma'], df),
        patterns=ms_pattern, free=True, argnums=1)

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = ms_pattern.fold(var_param)
        z = my_rs.randn(n_samples, dim)
        sqrtSigma = sqrtm(param_dict['Sigma'])
        return param_dict['mu'] + np.dot(z, sqrtSigma)/s[:,np.newaxis]

    def entropy(var_param):
        # ignore terms that depend only on df
        param_dict = ms_pattern.fold(var_param)
        return .5*np.log(np.linalg.det(param_dict['Sigma']))

    def mean_and_cov(var_param):
        param_dict = ms_pattern.fold(var_param)
        return param_dict['mu'], df / (df - 2.) * param_dict['Sigma']

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = ms_pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    return VariationalFamily(sample, entropy, logdensity, mean_and_cov,
                             pth_moment, ms_pattern.flat_length(True))


def black_box_klvi(var_family, logdensity, n_samples):
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        lower_bound = var_family.entropy(var_param) + np.mean(logdensity(samples))
        return -lower_bound

    objective_and_grad = value_and_grad(variational_objective)

    return objective_and_grad


def black_box_chivi(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)

    def objective_grad_and_log_norm(var_param):
        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm



def black_box_klvi_pd(var_family, logdensity, n_samples):
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        lower_bound = np.mean(logdensity(samples)) - np.mean(var_family.logdensity(samples, var_param))
        return -lower_bound

    objective_and_grad = value_and_grad(variational_objective)
    #objective_path_val = np.mean(logdensity(Samples))

    return objective_and_grad


def black_box_klvi_pd2(var_family, logdensity, n_samples):
    #a formulation which avoids path derivatives ...
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        a = partial(var_family.logdensity,var_param=var_param)
        def nested_fn(samples):
            lower_bound = np.mean(logdensity(samples)) - np.mean(a(samples))
            return -lower_bound
        b= nested_fn(samples)
        return b

    objective_and_grad = value_and_grad(variational_objective)
    #objective_path_val = np.mean(logdensity(Samples))

    return objective_and_grad


def _vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def _ensure_2d(a):
    while a.ndim < 2:
        a = a[:,np.newaxis]
    return a


def make_stan_log_density(fitobj):
    @primitive
    def log_density(x):
        return _vectorize_if_needed(fitobj.log_prob, x)
    def log_density_vjp(ans, x):
        return lambda g: _ensure_2d(g) * _vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(log_density, log_density_vjp)
    return log_density


def learning_rate_schedule(n_iters, learning_rate, learning_rate_end):
    if learning_rate <= 0:
        raise ValueError('learning rate must be positive')
    if learning_rate_end is not None:
        if learning_rate <= learning_rate_end:
            raise ValueError('initial learning rate must be greater than final learning rate')
        # constant learning rate for first quarter, then decay like a/(b + i)
        # for middle half, then constant for last quarter
        b = n_iters*learning_rate_end/(2*(learning_rate - learning_rate_end))
        a = learning_rate*b
        start_decrease_at = n_iters//4
        end_decrease_at = 3*n_iters//4
    for i in range(n_iters):
        if learning_rate_end is None or i < start_decrease_at:
            yield learning_rate
        elif i < end_decrease_at:
            yield a / (b + i - start_decrease_at + 1)
        else:
            yield learning_rate_end


def adagrad_optimize(n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                if has_log_norm:
                    obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                else:
                    obj_val, obj_grad = objective_and_grad(variational_param)
                    log_norm = 0
                value_history.append(obj_val)
                local_grad_history.append(obj_grad)
                local_log_norm_history.append(log_norm)
                log_norm_history.append(log_norm)
                if len(local_grad_history) > window:
                    local_grad_history.pop(0)
                    local_log_norm_history.pop(0)
                grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                scaled_grads = grad_scale[:,np.newaxis]*np.array(local_grad_history)
                accum_sum = np.sum(scaled_grads**2, axis=0)
                variational_param = variational_param - curr_learning_rate*obj_grad/np.sqrt(epsilon + accum_sum)
                if i >= 3*n_iters // 4:
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
    variational_param_history = np.array(variational_param_history)
    smoothed_opt_param = np.mean(variational_param_history, axis=0)
    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history))


def rmsprop_IA_optimize_with_rhat(n_iters, objective_and_grad, init_param,K,
                        has_log_norm=False, window=500, learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                                  r_mean_threshold=1.15, r_sigma_threshold=1.20,learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    sum_grad_norm = 0.
    alpha = 0.9
    scaled_sum_grad_norm = 0.
    variational_param_history_list = []
    averaged_variational_param_history_list = []
    variational_param_list = []
    averaged_variational_param_list = []
    #window_size=500

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.2
        #variational_param = init_param
        #print(variational_param)
        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    prev_variational_param = variational_param
                    if has_log_norm:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0
                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)

                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        # sum_grad_squared=obj_grad**2
                        sum_grad_squared = grad_norm
                    else:
                        # sum_grad_squared = sum_grad_squared*alpha + (1.-alpha)*obj_grad**2
                        sum_grad_squared = grad_norm * alpha + (1. - alpha) * grad_norm
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    scaled_sum_grad_norm = scaled_sum_grad_norm * alpha + (1 - alpha) * accum_sum
                    old_variational_param = variational_param.copy()
                    variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(
                        epsilon + sum_grad_squared)
                    # variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(epsilon + scaled_sum_grad_norm)
                    variational_param_history.append(old_variational_param)
                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))


            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                #pass
                progress.close()

        variational_param_history_array = np.array(variational_param_history)
        variational_param_history_list.append(variational_param_history_array)
        variational_param_list.append(variational_param)

    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
    rhats = compute_R_hat_adaptive_numpy(variational_param_history_chains, window_size=rhat_window)
    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]

    start_swa_m_iters = n_iters - 2000
    start_swa_s_iters = start_swa_m_iters
    for ee, w in enumerate(rhat_mean_windows):
        if ee == (rhat_mean_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_mean_windows[ee] < r_mean_threshold).all() and (rhat_mean_windows[ee + 1] < r_mean_threshold).all():
            start_swa_m_iters = ee * rhat_window
            break


    for ee, w in enumerate(rhat_sigma_windows):
        if ee == (rhat_sigma_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_sigma_windows[ee] < r_sigma_threshold).all() and (rhat_sigma_windows[ee + 1] < r_sigma_threshold).all():
            start_swa_s_iters = ee * rhat_window
            break

    optimisation_log = dict()

    for o in range(n_optimisers):
        q_locs_dim = variational_param_history_chains[o,:,:K]
        q_log_sigmas_dim = variational_param_history_chains[o, :, K:]
        q_swa_means_iters, q_swa_mean = stochastic_iterate_averaging(q_locs_dim,
                                                                    start_swa_m_iters)
        q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
                                                                              start_swa_m_iters)
        averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))
        averaged_variational_param_list.append(averaged_variational_params)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters
    optimisation_log['r_hat_mean'] = rhat_mean_windows
    optimisation_log['r_hat_sigma'] = rhat_sigma_windows

    return (variational_param, variational_param_history_chains, averaged_variational_param_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)



def adam_IA_optimize_with_rhat(n_iters, objective_and_grad, init_param,K,
                        has_log_norm=False, window=500,  learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                               r_mean_threshold=1.15, r_sigma_threshold=1.20, learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    sum_grad_norm = 0.
    alpha = 0.9
    scaled_sum_grad_norm = 0.
    variational_param_history_list = []
    averaged_variational_param_history_list = []
    variational_param_list = []
    averaged_variational_param_list = []
    window_size=500
    grad_val= 0.
    grad_squared=0
    beta1=0.9
    beta2=0.999
    r_mean_threshold= 1.10
    r_sigma_threshold = 1.20

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.2
        #variational_param = init_param
        #print(variational_param)
        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    prev_variational_param = variational_param
                    if has_log_norm:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0
                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)

                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        grad_squared = 0.9 * obj_grad ** 2
                        grad_val = 0.9 * obj_grad
                    else:
                        grad_squared = grad_squared * beta2 + (1. - beta2) * obj_grad ** 2
                        grad_val = grad_val * beta1 + (1. - beta1) * obj_grad
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    old_variational_param = variational_param.copy()
                    m_hat = grad_val / (1 - np.power(beta1, i + 2))
                    v_hat = grad_squared / (1 - np.power(beta2, i + 2))
                    variational_param = variational_param - curr_learning_rate * m_hat / np.sqrt(epsilon + v_hat)
                    if averaging is True and i > start_avg_iter:
                        averaged_variational_param = (variational_param + old_variational_param * (
                                    i - start_avg_iter)) / (i - start_avg_iter + 1)
                        averaged_variational_param_history.append(averaged_variational_param)
                    variational_param_history.append(old_variational_param)
                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))


            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                #pass
                progress.close()

        variational_param_history_array = np.array(variational_param_history)
        variational_param_history_list.append(variational_param_history_array)
        variational_param_list.append(variational_param)

    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
    rhats = compute_R_hat_adaptive_numpy(variational_param_history_chains, window_size=rhat_window)
    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]

    start_swa_m_iters = n_iters - 2000
    start_swa_s_iters = n_iters - 2000
    for ee, w in enumerate(rhat_mean_windows):
        if ee == (rhat_mean_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_mean_windows[ee] < r_mean_threshold).all() and (rhat_mean_windows[ee + 1] < r_mean_threshold).all():
            start_swa_m_iters = ee * rhat_window
            break

    for ee, w in enumerate(rhat_sigma_windows):
        if ee == (rhat_sigma_windows.shape[0] - 1):
            continue
    # print(R_hat_window_np[ee])
        if (rhat_sigma_windows[ee] < r_sigma_threshold).all() and (rhat_sigma_windows[ee + 1] < r_sigma_threshold).all():
            start_swa_s_iters = ee * rhat_window
            break

    optimisation_log = dict()

    for o in range(n_optimisers):
        q_locs_dim = variational_param_history_chains[o,:,:K]
        q_log_sigmas_dim = variational_param_history_chains[o, :, K:]
        q_swa_means_iters, q_swa_mean = stochastic_iterate_averaging(q_locs_dim,
                                                                    start_swa_m_iters)
        q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
                                                                              start_swa_m_iters)
        averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))
        averaged_variational_param_list.append(averaged_variational_params)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters

    optimisation_log['r_hat_mean'] = rhat_mean_windows
    optimisation_log['r_hat_sigma'] = rhat_sigma_windows

    return (variational_param, variational_param_history_chains, averaged_variational_param_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)

