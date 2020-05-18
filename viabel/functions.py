
#import numpy as np
import autograd.numpy as np
from autograd.extend import primitive


from psis import psislw

def compute_R_hat(chains, warmup=500):
    #first axis is relaisations, second is iters
    # N_realisations X N_iters X Ndims
    jitter = 1e-8
    chains = chains[:, warmup:, :]
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    K = chains.shape[2]
    if n_iters%2 == 1:
        n_iters = int(n_iters - 1)
        chains = chains[:,:n_iters-1,:]

    n_iters = n_iters // 2
    psi = np.reshape(chains, (n_chains * 2, n_iters, K))
    n_chains2 = n_chains*2
    psi_dot_j = np.mean(psi, axis=1)
    psi_dot_dot = np.mean(psi_dot_j, axis=0)
    s_j_2 = np.sum((psi - np.expand_dims(psi_dot_j, axis=1)) ** 2, axis=1) / (n_iters - 1)
    B = n_iters * np.sum((psi_dot_j - psi_dot_dot) ** 2, axis=0) / (n_chains2 - 1)
    W = np.nanmean(s_j_2, axis=0)
    W = W + jitter
    var_hat = (n_iters - 1) / n_iters + (B / (n_iters*W))
    R_hat = np.sqrt(var_hat)
    return var_hat, R_hat

# def compute_R_hat_adaptive_numpy(chains, window_size=100):
#     # numpy function for computing R-hat , maybe inefficient but does the job ...
#     n_chains, n_iters, K = chains.shape
#     n_batches = n_iters//window_size
#     R_hat_adaptive = np.zeros((n_batches,K))
#
#     for i in range(int(n_batches)):
#         varhat, rhat = compute_R_hat(chains[:, :n_iters-(i+1)*window_size:-1,:], warmup=0)
#         R_hat_adaptive[i, :] = rhat
#     return R_hat_adaptive

def compute_R_hat_adaptive_numpy(chains, window_size=100):
    # numpy function for computing R-hat , maybe inefficient but does the job ...
    n_chains, n_iters, K = chains.shape
    n_windows = n_iters//window_size
    chains_reshaped = np.transpose(np.reshape(chains, [n_chains, n_windows,
                                                       window_size, -1]),
                                     [1, 0, 2, 3])
    r_hats = np.array([compute_R_hat(chains_reshaped[i, :], warmup=0)[1] for i in range(chains_reshaped.shape[0])])
    return r_hats

def compute_R_hat_halfway(chains, interval=100, start=1000):
    n_chains, n_iters, K= chains.shape
    n_subchains = n_iters //interval
    r_hats_halfway = list()

    for i in range(n_subchains):
        sub_chains = chains[:, :start+(i+1)*interval,:]
        n_sub_chains, n_sub_iters, K = sub_chains.shape
        r_hat_current = compute_R_hat(sub_chains, warmup=n_sub_iters//2)[1]
        r_hats_halfway.append(r_hat_current)

    return np.array(r_hats_halfway)


def compute_R_hat_halfway_light(chains, interval=100, start=1000):
    n_chains, n_iters, K= chains.shape
    n_subchains = n_iters //interval
    r_hats_halfway = list()
    r_hat_current = compute_R_hat(chains, warmup=n_iters//2)[1]
    #r_hats_halfway.append(r_hat_current)
    #return np.array(r_hats_halfway)
    return r_hat_current


def stochastic_iterate_averaging(estimate, start):
    N = estimate.shape[0]
    if N - start <= 0:
        raise "Start of stationary distribution must be lower than number of iterates"

    window_lengths = np.reshape(np.arange(start, N) - start + 1,
                                [-1, 1])
    estimate_iters = np.cumsum(estimate[start:,:], axis=0) / window_lengths
    estimate_mean = estimate_iters[-1]
    return (estimate_iters, estimate_mean)


def safe_root(N):
    i = np.sqrt(N)
    j = int(i)
    if i != j:
        raise ValueError("N is not square!")
    return j


def _sym_index(k1, k2):
    def ld_ind(k1, k2):
        return int(k2 + k1 * (k1 + 1) / 2)

    if k2 <= k1:
        return ld_ind(k1, k2)
    else:
        return ld_ind(k2, k1)

def _vectorize_ld_matrix(mat):
    nrow, ncol = np.shape(mat)
    if nrow != ncol:
        raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


@primitive
def flat_to_triang(flat_mat):
    N = flat_mat.size
    M = int(-1 + np.sqrt(8*N+1))//2
    ret = np.zeros(( M, M))

    count = 0
    for m in range(M):
        for mm in range(m+1):
            #print(count)
            #flat_mat[count])
            ret[m, mm] = flat_mat[count]
            count = count+1
    return ret


def flat_to_triang_vjp(g):
    assert g.shape[0] == g.shape[1]
    return _vectorize_ld_matrix(g)


@primitive
def triang_to_flat(L):
    D, _, M = L.shape
    N = M*(M+1)//2
    flat = np.empty((N, D))
    for d in range(D):
        count = 0;
        for m in range(M):
            for mm in range(m+1):
                flat[count,d] = L[d, m, mm]
                count = count +1
    return flat


def compute_posterior_moments(prior_mean, prior_covariance, noise_variance, x, y):
    prior_L = np.linalg.cholesky(prior_covariance)
    inv_L = np.linalg.inv(prior_L)
    prior_precision = inv_L.T@inv_L
    S_precision = prior_precision + x.T @ x *(1. / noise_variance)
    a = np.linalg.cholesky(S_precision)
    tmp1 = np.linalg.inv(a)
    S = tmp1.T @ tmp1
    post_S=S
    post_mu = prior_precision@prior_mean + (1./noise_variance)* x.T@ y
    post_mu = post_S@ post_mu
    return post_mu, post_S



def get_samples_and_log_weights(logdensity, var_family, var_param, n_samples):
    samples = var_family.sample(var_param, n_samples)
    log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
    return samples, log_weights



def psis_correction(logdensity, var_family, var_param, n_samples):
    samples, log_weights = get_samples_and_log_weights(logdensity, var_family,
                                                       var_param, n_samples)
    smoothed_log_weights, khat = psislw(log_weights)
    return samples.T, smoothed_log_weights, khat

