
import numpy as np

def compute_R_hat(chains, warmup=500):
    #first axis is relaisations, second is iters
    # N_realisations X N_iters X Ndims
    jitter = 1e-8
    #print(chains.shape)
    chains = chains[:, warmup:, :]
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    K = chains.shape[2]
    #print(warmup)
    print(chains.shape)

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
    #rhats = np.zeros((n_windows,K))

    chains_reshaped = np.transpose(np.reshape(chains, [n_chains, n_windows,
                                                       window_size, -1]),
                                     [1, 0, 2, 3])
    print(chains_reshaped.shape)
    #r_hats = np.apply_over_axes(lambda chains: compute_R_hat(chains, warmup=0)[1],0,
    #                   chains_reshaped)

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

    return r_hats_halfway


def stochastic_weight_averaging(estimate, start):
    N = estimate.shape[0]
    if N - start <= 0:
        raise "Start of stationary distribution must be lower than number of iterates"

    window_lengths = np.reshape(np.arange(start, N) - start + 1,
                                [-1, 1])
    estimate_iters = np.cumsum(estimate[start:,:], axis=0) / window_lengths
    estimate_mean = estimate_iters[-1]
    return (estimate_iters, estimate_mean)