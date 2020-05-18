
import numpy as np
import random

logit = lambda x: 1./ (1 +np.exp(-x))

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

def data_generator_logistic(N, K, alpha=1., noise_variance=0.4, rho=0.0, seed=100):
    noise_sigma= np.sqrt(noise_variance)
    #regression_data = data_generator_linear1(N, K, alpha, noise_variance, rho=rho, seed=seed)
    regression_data = data_generator_linear(N, K, noise_sigma, rho=rho, seed=seed)
    Y_linear = regression_data['Y']
    p_full = logit(Y_linear)
    y_full = np.random.binomial(n=1, p=p_full)
    regression_data['Y'] = y_full
    return regression_data