
#import numpy as np
import autograd.numpy as np
from autograd.extend import primitive
import scipy

import arviz



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


# computes posterior moments for Linear Regression with known variance- analytically
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


def compute_elpd(posterior_mean, posterior_covariance, X_test, Y_test, seed=42, n_samples=50000):
    np.random.seed(seed=seed)
    w_samples = np.random.multivariate_normal(posterior_mean, posterior_covariance, n_samples)
    Y_preds = X_test@ w_samples.T
    q_theta_s = scipy.stats.multivariate_normal.pdf(w_samples, posterior_mean, posterior_covariance)
    likvals = np.zeros((100, n_samples))

    liks = scipy.stats.norm.logpdf(Y_test, Y_preds.T, 0.5)
    vals = liks.T@q_theta_s
    vals2 = liks.sum()/n_samples
    #return  vals.sum()/n_samples
    return vals2


def compute_elpd_general(logdensity, var_family, var_param, n_samples, seed=42):
    np.random.seed(seed=seed)
    samples = var_family.sample(var_param, n_samples)
    liks = logdensity(samples)
    vals2 = liks.sum() / n_samples
    return vals2


def get_samples_and_log_weights(logdensity, var_family, var_param, n_samples):
    samples = var_family.sample(var_param, n_samples)
    log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
    return samples, log_weights



def psis_correction(logdensity, var_family, var_param, n_samples):
    samples, log_weights = get_samples_and_log_weights(logdensity, var_family,
                                                       var_param, n_samples)
    smoothed_log_weights, khat = psislw(log_weights)
    return samples.T, smoothed_log_weights, khat

