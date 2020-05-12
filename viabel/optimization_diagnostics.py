
import autograd.numpy as np
from autograd.extend import primitive

from .functions import compute_R_hat, compute_R_hat_halfway



def monte_carlo_se(iterate_chains, warmup=500):

    chains = iterate_chains[:, warmup:, :]
    n_chains, N_iters, K = chains.shape[0], chains.shape[1], chains.shape[2]
    mcse_combined_list = np.zeros((N_iters,K))
    print(chains.shape)
    for i in range(1,N_iters):
        chains_sub = chains[:, :i,:]
        n_chains, n_iters, K = chains_sub.shape[0], chains_sub.shape[1], chains_sub.shape[2]
        chains_flat = np.reshape(chains_sub, (n_chains*i, K))
        variances = np.var(chains, ddof=1, axis=1)
        variances_combined = np.var(chains_flat, ddof=1, axis=0)
        mean_var_chains = np.mean(variances, axis=0)
        mcse_per_chain = np.sqrt(variances / n_iters)
        mcse_combined = np.sqrt(variances_combined/(i*n_chains))
        mcse_combined_list[i] = mcse_combined

    #print(mcse_per_chain.shape)
    return mcse_per_chain, mcse_combined_list


# compute mcmcse for a chain/array
def monte_carlo_se2(iterate_chains, warmup=500, param_idx=0):
    '''
    compute monte carlo standard error for all chains and one parameter at a time.
    :param iterate_chains:
    :param warmup: warmup iterates
    :return: array of mcse error for all chains for a particular variational parameter.
    '''
    chains = iterate_chains[:, warmup:, param_idx]
    n_chains, N_iters = chains.shape[0], chains.shape[1]
    mcse_combined_list = np.zeros(N_iters)
    Neff, _, _, _ = autocorrelation(iterate_chains, warmup=0, param_idx=param_idx)

    for i in range(100, N_iters):
        chains_sub = chains[:, :i]
        n_chains, n_iters = chains_sub.shape[0], chains_sub.shape[1]
        chains_flat = np.reshape(chains_sub, (n_chains*i, 1))
        variances_combined = np.var(chains_flat, ddof=1, axis=0)
        Neff , _, _, _ = autocorrelation(iterate_chains[:,:i,:], warmup=0, param_idx=param_idx)
        mcse_combined = np.sqrt(variances_combined/Neff)
        mcse_combined_list[i] = mcse_combined
    #print(mcse_combined_list[:20,:5])
    return  np.array(mcse_combined_list)


def montecarlo_se(iterate_chains, warmup=500):
    '''
    compute monte carlo standard error for all chains and all parameters at once.
    :param iterate_chains:
    :param warmup:
    :return: array of mcse error for using all chains at once and for all parameters.
    '''
    n_chains, N_iters, K = iterate_chains.shape[0], iterate_chains.shape[1], iterate_chains.shape[2]
    chains_flat = np.reshape(iterate_chains, (n_chains * N_iters, K))
    variances_combined = np.var(chains_flat, ddof=1, axis=0)
    #mcse_combined = np.sqrt(variances_combined / Neff)

    Neff = np.zeros(K)
    for pmx in range(K):
        chains = iterate_chains[:, warmup:, pmx]
        #print(chains_flat.shape)
        a, _, _, _ = autocorrelation(iterate_chains, warmup=0, param_idx=pmx)
        Neff[pmx] = a

    mcse_combined = np.sqrt(variances_combined / Neff)
    return mcse_combined


def acf(iterate_chains, warmup=500, param_idx=0):
    pass


def autocorrelation(iterate_chains, warmup=500, param_idx=0, lag_max=80):
    '''
    computes autocorrelation using ALL chains for a particular variational parameter.
    :param iterate_chains:
    :param warmup:
    :param param_idx:
    :param lag_max:
    :return:
    '''
    chains = iterate_chains[:, warmup:, param_idx]
    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    if n_chains == 1:
        var_between = 0
    else:
        var_between = n_iters * np.var(means, ddof=1)

    var_chains = np.mean(variances, axis=0)
    var_pooled = ((n_iters - 1.) * var_chains + var_between) /n_iters
    n_pad = int(2**np.ceil(1. + np.log2(n_iters)))
    freqs =   np.fft.rfft(chains - np.expand_dims(means, axis=1), n_pad)
    autocov = np.fft.irfft(np.abs(freqs)**2)[:,:n_iters].real
    autocov= autocov / np.arange(n_iters, 0, -1)
    rho_t = 0
    lag = 1
    a = []
    neff_array = []
    while lag < 100:
        val =   1. - (var_chains - np.mean(autocov[:,lag])) / var_pooled
        a.append(val)
        if val >= 0:
            rho_t = rho_t + val
        else:
            #break
            rho_t =rho_t
        lag = lag + 1

    neff = n_iters *n_chains /(1 + 2*rho_t)
    return neff, rho_t, autocov, np.asarray(a)


def compute_khat_iterates(iterate_chains, warmup=500, param_idx=0, increasing= True, fraction=0.05):
    '''
    function computes the khat for iterates of VI, preferable to run it after approximate convergence .
    :param iterate_chains:
    :param warmup:
    :param param_idx:
    :param increasing:
    :param fraction:
    :return:
    '''
    chains = iterate_chains[:, warmup:, param_idx]
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    #fraction = 0.05

    k_hat_values = np.zeros(n_chains)
    for i in range(n_chains):
        if increasing:
            sorted_chain = np.sort(chains[i,:])
        else:
            sorted_chain = np.sort(-chains[i,:])

        ind_last = int(n_iters * fraction)
        filtered_chain = sorted_chain[n_iters-ind_last:]
        if increasing:
            filtered_chain = filtered_chain -np.min(filtered_chain)
        else:
            filtered_chain = filtered_chain -np.min(filtered_chain)
        k_post, _ = gpdfit(filtered_chain)
        k_hat_values[i] = k_post

    return np.nanmax(k_hat_values)




# taken from arviz ...
def gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).
    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.
    Parameters
    ----------
    ary : array
        sorted 1D data array
    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    sigma = -k_post / b_post

    return k_post, sigma