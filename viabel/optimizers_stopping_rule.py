
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm

from functools import partial

import tqdm
import scipy.stats as stats
from  .optimization_diagnostics import autocorrelation, monte_carlo_se, monte_carlo_se2, compute_khat_iterates, \
    gpdfit, montecarlo_se
from .functions import compute_R_hat, compute_R_hat_adaptive_numpy, compute_R_hat_halfway, stochastic_iterate_averaging

from .functions import flat_to_triang, triang_to_flat

__all__ = [
    'adagrad_optimize',
    'rmsprop_IA_optimize_with_rhat',
    'adam_IA_optimize_with_rhat'
]



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


def adagrad_optimize_stop(n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, tolerance=0.01, eval_elbo=100,  stopping_rule=1, learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    pareto_k_list = []
    neff_list = []
    diff_val = []
    prev_elbo = 0.
    elbo_diff_rel_list = []
    pmz_size = init_param.size
    mcse_all = np.zeros((pmz_size,1))
    elbo_diff_rel_med= 10.
    elbo_diff_rel_avg = 10.

    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                if has_log_norm == 1:
                    obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                elif has_log_norm == 2:
                    obj_val, obj_grad, paretok, neff = objective_and_grad(variational_param)
                    log_norm = 0.
                    if paretok > 0.25:
                        pareto_k_list.append(paretok)
                        neff_list.append(neff)
                else:
                    obj_val, obj_grad = objective_and_grad(variational_param)
                    log_norm= 0.


                if stopping_rule==1 and i > 600 and i % eval_elbo == 0:
                    print(i)
                    elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo+1e-8)
                    elbo_diff_rel_list.append(elbo_diff_rel)
                    elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                    elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)
                    print(elbo_diff_rel_med)
                    print(elbo_diff_rel_avg)

                prev_elbo = obj_val


                start_stats = 500
                mcse_se_combined_list = np.zeros((pmz_size,1))
                if stopping_rule == 2 and i > 600 and i % eval_elbo == 0:
                        #neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats, i)
                    mcse_se_combined_list = montecarlo_se(np.array(variational_param_history)[None,:], 0)
                    print(mcse_se_combined_list.shape)
                    print(mcse_all.shape)
                    print(np.min(mcse_all[:,-1]))
                    mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:,None]))


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
                if i >= 50:
                    variational_param_history.append(variational_param.copy())

                if i % 10 == 0:
                    avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))

                if stopping_rule==1 and elbo_diff_rel_med  <= tolerance:
                    print('yay median')
                    break
                if stopping_rule == 1 and elbo_diff_rel_avg <= tolerance:
                    print('yay mean')
                    break

                #print(np.max(mcse_all[:,-1]))
                if stopping_rule ==2 and i > 700 and i % eval_elbo == 0 and (np.min(mcse_all[:,-1]) <= 0.001 ):
                    print(mcse_all.shape)
                    print(mcse_all[:,-1])
                    print('lol')
                    break
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            progress.close()
        finally:
            progress.close()


    variational_param_history = np.array(variational_param_history)
    variational_param_history_list=[]
    variational_param_history_list.append(variational_param_history)
    variational_param_history_chains = np.stack(variational_param_history_list, axis=0)

    pmz_size = variational_param_history_chains.shape[1]
    optimisation_log = dict()

    khat_iterates = []
    khat_iterates2 = []
    start_stats = n_iters - 3000

    #for i in range(pmz_size):
    #    khat_i = compute_khat_iterates(variational_param_history_chains, 0, i, increasing=True)
    #    khat_iterates.append(khat_i)

    #for j in range(pmz_size):
    #    khat_i = compute_khat_iterates(variational_param_history_chains, 0, j, increasing=False)
    #    khat_iterates2.append(khat_i)


    khat_objective,_ = gpdfit(np.array(value_history))
    value_history_neg = [-a for a in value_history]

    khat_objective2, _ = gpdfit(np.array(value_history_neg))
    khat_iterates.append(khat_objective)
    khat_iterates2.append(khat_objective2)
    #khat_iterates.append(khat_objective)

    khat_iterates_array = np.stack(khat_iterates, axis=0)
    khat_iterates_array2 = np.stack(khat_iterates2, axis=0)
    optimisation_log['khat_iterates2'] = khat_iterates_array2
    optimisation_log['khat_iterates'] = khat_iterates_array
    optimisation_log['khat_objective'] = khat_objective
    if has_log_norm == 2:
        optimisation_log['paretok'] = np.array(pareto_k_list)
        optimisation_log['neff'] = np.array(neff_list)

    smoothed_opt_param = np.mean(variational_param_history, axis=0)
    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history), optimisation_log)



def adam_IA_optimize_stop(n_iters, objective_and_grad, init_param, K,
                        has_log_norm=False, window=500,  learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                               r_mean_threshold=1.15, r_sigma_threshold=1.20, tail_avg_iters=2000,
                               eval_elbo=100, tolerance=0.01, stopping_rule=1,
                               learning_rate_end=None):
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
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    window_size=500
    grad_val= 0.
    grad_squared=0
    beta1=0.9
    beta2=0.999
    prev_elbo = 0.
    elbo_diff_rel_list = []
    pmz_size = init_param.size
    mcse_all = np.zeros(pmz_size)
    elbo_diff_rel_med= 10.
    elbo_diff_rel_avg = 10.

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.2
        #variational_param = init_param
        #print(variational_param)
        total_iters=0
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

                    if stopping_rule == 1 and i > 700 and i % eval_elbo == 0:
                        print(i)
                        elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo + 1e-8)
                        elbo_diff_rel_list.append(elbo_diff_rel)
                        elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                        elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)
                        print(elbo_diff_rel_med)
                        print(elbo_diff_rel_avg)

                    prev_elbo = obj_val
                    total_iters = i

                    mcse_se_combined_list = np.zeros((pmz_size, 1))
                    if stopping_rule == 2 and i > 600 and i % eval_elbo == 0:
                        # neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats, i)
                        mcse_se_combined_list = montecarlo_se(np.array(variational_param_history)[None, :], 0)
                        print(np.min(mcse_all[:, -1]))
                        mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:, None]))

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

                    if i > 100:
                        variational_param_history.append(old_variational_param)

                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))

                    if stopping_rule == 1 and elbo_diff_rel_med <= tolerance:
                        print('yay median')
                        break
                    if stopping_rule and elbo_diff_rel_avg <= tolerance:
                        print('yay mean')
                        break

                    if stopping_rule == 2 and i > 700 and i % eval_elbo == 0 and (np.median(mcse_all[:, -1]) <= 0.02):
                        print('lol')
                        break


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

    rhats_halfway = compute_R_hat_halfway(variational_param_history_chains, interval=100, start=200)

    rhat_mean_windows, rhat_sigma_windows = rhats[:,:K], rhats[:,K:]
    rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway[:, :K], rhats_halfway[:, K:]

    start_swa_m_iters = total_iters - tail_avg_iters
    start_swa_s_iters = total_iters - tail_avg_iters
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
                                                                              start_swa_s_iters)

        #averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))



        #q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
        #                                                                      start_swa_s_iters)

        #if start_swa_s_iters > start_swa_m_iters:
        #    averaged_variational_params = np.hstack((q_swa_means_iters[start_swa_s_iters-start_swa_m_iters:], q_swa_log_sigmas_iters))
        #else:
        #    averaged_variational_params = np.hstack(
        #        (q_swa_means_iters, q_swa_log_sigmas_iters[start_swa_m_iters-start_swa_s_iters:]))
        #averaged_variational_param_list.append(averaged_variational_params)
        averaged_variational_mean_list.append(q_swa_means_iters)
        averaged_variational_sigmas_list.append(q_swa_log_sigmas_iters)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters

    optimisation_log['r_hat_mean'] = rhat_mean_windows
    optimisation_log['r_hat_sigma'] = rhat_sigma_windows

    optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
    optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway
    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)




def rmsprop_IA_optimize_stop(n_iters, objective_and_grad, init_param,K,
                        has_log_norm=False, window=500, learning_rate=.01,
                        epsilon=.000001, rhat_window=500, averaging=True, n_optimisers=1,
                        r_mean_threshold=1.15, r_sigma_threshold=1.20, tail_avg_iters=1000,
                        eval_elbo=100, tolerance=0.01, stopping_rule=1, avg_grad_norm=False,
                        learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    sum_grad_norm = 0.
    alpha = 0.99
    scaled_sum_grad_norm = 0.
    variational_param_history_list = []
    averaged_variational_param_history_list = []
    variational_param_list = []
    averaged_variational_param_list = []
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    pareto_k_list = []
    neff_list = []
    prev_elbo = 0.
    elbo_diff_rel_list = []
    pmz_size = init_param.size
    mcse_all = np.zeros((pmz_size,1))
    elbo_diff_rel_med= 10.
    elbo_diff_rel_avg = 10.
    #window_size=500
    S=0


    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.5
        #variational_param = init_param
        #print(variational_param)
        #actual number of iterations
        total_iters=0
        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    prev_variational_param = variational_param
                    if has_log_norm == 1:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    elif has_log_norm ==2:
                        obj_val, obj_grad, paretok ,neff = objective_and_grad(variational_param)
                        log_norm=0.
                        if paretok > 0.25:
                            pareto_k_list.append(paretok)
                            neff_list.append(neff)
                    elif has_log_norm == 3:
                        obj_val, obj_grad, S, paretok, neff = objective_and_grad(variational_param, S)
                        log_norm=0.
                        if paretok > 0.06:
                            pareto_k_list.append(paretok)
                            neff_list.append(neff[0])
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0

                    total_iters = i
                    if i > 700 and i % eval_elbo == 0:
                        print(i)
                        elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo + 1e-8)
                        elbo_diff_rel_list.append(elbo_diff_rel)
                        elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                        elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)
                        print(elbo_diff_rel_med)
                        print(elbo_diff_rel_avg)

                    prev_elbo = obj_val
                    total_iters = i

                    mcse_se_combined_list = np.zeros((pmz_size, 1))
                    if stopping_rule == 2 and i > 600 and i % eval_elbo == 0:
                        mcse_se_combined_list = montecarlo_se(np.array(variational_param_history)[None, :], 0)
                        print(mcse_se_combined_list.shape)
                        print(mcse_all.shape)
                        print(np.min(mcse_all[:, -1]))
                        mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:, None]))
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
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm
                        else:
                            sum_grad_squared=obj_grad**2
                        #sum_grad_squared = grad_norm
                    else:
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm * alpha + (1. - alpha) * grad_norm
                        else:
                            sum_grad_squared = sum_grad_squared*alpha + (1.-alpha)*obj_grad**2
                        #sum_grad_squared = grad_norm * alpha + (1. - alpha) * grad_norm
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    scaled_sum_grad_norm = scaled_sum_grad_norm * alpha + (1 - alpha) * accum_sum
                    old_variational_param = variational_param.copy()
                    variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(
                        epsilon + sum_grad_squared)
                    # variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(epsilon + scaled_sum_grad_norm)
                    if i > 100:
                        variational_param_history.append(old_variational_param)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))

                    if stopping_rule == 1 and elbo_diff_rel_med <= tolerance:
                        print('yay median')
                        break
                    if stopping_rule == 1 and elbo_diff_rel_avg <= tolerance:
                        print('yay mean')
                        break

                    if stopping_rule == 2 and i > 700 and i % eval_elbo == 0 and (np.median(mcse_all[:, -1]) <= 0.02):
                        print('lol')
                        break

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

    start_swa_m_iters = total_iters - tail_avg_iters
    start_swa_s_iters = start_swa_m_iters

    optimisation_log = dict()
    start_stats = max(start_swa_m_iters, start_swa_s_iters)
    print('start stats:')
    # compute autocorrelation for the chains
    #neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats)
    # compute monte carlo se for chains ..
    #mcse_per_chain, mcse_combined_list =  monte_carlo_se(variational_param_history_chains, start_stats)
    #pmz_size = variational_param_history_chains.shape[2]
    #Neff = np.zeros(pmz_size)
    #Rhot = []

    #mcmc_se2= []
    #khat_iterates =[]
    #khat_iterates_min =[]
    #khat_iterates2 = []

    # for i in range(pmz_size):
    #     neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_history_chains, start_stats, i)
    #     #mcse_se_combined = monte_carlo_se2(variational_param_history_chains, start_stats,i)
    #     Neff[i] = neff
    #     #mcmc_se2.append(mcse_se_combined)
    #     Rhot.append(rho_t)
    #     khat_i = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=True)
    #     khat_iterates.append(khat_i)
    #
    #
    # for i in range(pmz_size):
    #     khat_i2 = compute_khat_iterates(variational_param_history_chains, start_stats, i, increasing=False)
    #     khat_iterates2.append(khat_i2)
    #
    #
    # khat_objective,_ = gpdfit(np.array(value_history))
    # khat_iterates.append(khat_objective)
    # value_history_neg = [-a for a in value_history]
    #
    # khat_objective2, _ = gpdfit(np.array(value_history_neg))
    # khat_iterates2.append(khat_objective2)
    #
    # rhot_array = np.stack(Rhot, axis=0)
    # khat_iterates_array = np.stack(khat_iterates, axis=0)
    # khat_iterates_array2 = np.stack(khat_iterates2, axis=0)
    #
    #
    # if has_log_norm == 2 or has_log_norm == 3:
    #     optimisation_log['paretok'] = pareto_k_list
    #     optimisation_log['neff'] = Neff

    #mcmc_se2_array = np.stack(mcmc_se2, axis=1)
    for o in range(n_optimisers):
        q_locs_dim = variational_param_history_chains[o,:,:K]
        q_log_sigmas_dim = variational_param_history_chains[o, :, K:]

        #print(start_swa_m_iters)
        #print(q_locs_dim.shape)
        #print(q_log_sigmas_dim.shape)
        q_swa_means_iters, q_swa_mean = stochastic_iterate_averaging(q_locs_dim,
                                                                    start_swa_m_iters)
        q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
                                                                              start_swa_s_iters)
        #averaged_variational_params = np.hstack((q_swa_means_iters, q_swa_log_sigmas_iters))
        #q_swa_log_sigmas_iters, q_swa_log_sigma = stochastic_iterate_averaging(q_log_sigmas_dim,
        #                                                                      start_swa_s_iters)
        #if start_swa_s_iters > start_swa_m_iters:
        #    averaged_variational_params = np.hstack((q_swa_means_iters[start_swa_s_iters-start_swa_m_iters:], q_swa_log_sigmas_iters))
        #else:
        #    averaged_variational_params = np.hstack(
        #        (q_swa_means_iters, q_swa_log_sigmas_iters[start_swa_m_iters-start_swa_s_iters:]))
        #averaged_variational_param_list.append(averaged_variational_params)
        averaged_variational_mean_list.append(q_swa_means_iters)
        averaged_variational_sigmas_list.append(q_swa_log_sigmas_iters)

    optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
    optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters

    # optimisation_log['neff'] = Neff
    # optimisation_log['autocov'] = autocov
    # optimisation_log['rhot'] = rhot_array
    # optimisation_log['start_stats'] = start_stats
    # optimisation_log['mcmc_se'] = mcse_combined_list
    # #optimisation_log['mcmc_se2'] = mcmc_se2_array
    #
    # optimisation_log['khat_iterates'] = khat_iterates_array
    # optimisation_log['khat_iterates2'] = khat_iterates_array2

    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)
