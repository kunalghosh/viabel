
import autograd.numpy as np

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


def data_generator_linear(N, K, alpha=1., noise_variance=1., rho=0.0, seed=0):
    np.random.seed(seed=seed)
    mean_val = 0.
    noise_sigma = np.sqrt(noise_variance)
    M = 1
    alpha_I = alpha*np.eye(K)
    X_mean = np.zeros(K)
    K_mat = np.zeros((K,K))

    for i in range(K_mat.shape[0]):
        for j in range(K_mat.shape[1]):
            K_mat[i,j] = rho**np.abs(i-j)

    X = np.random.multivariate_normal(X_mean, K_mat, (N,))
    beta = np.random.multivariate_normal(X_mean, alpha_I, (1,)).T
    y_mean = X @ beta
    Y = y_mean + np.random.multivariate_normal(np.array([0.]),
                                               np.eye(1) * noise_variance, (N,))
    regression_data = {}
    regression_data['X'] = X
    regression_data['Y'] = Y
    regression_data['W'] = beta
    return regression_data