
import autograd.numpy as np
from autograd.extend import primitive
import scipy

import  matplotlib.pyplot as plt

#from  .optimization_diagnostics import  monte_carlo_se
from scipy.stats import t

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


df_list = np.array([1,2,4,8,16])
n_samples= 100000
fraction = 0.1
K_list = []

for i in range(df_list.size):
    df = df_list[i]
    samples = t.rvs(df, size=n_samples)

    samples_sorted = np.sort(samples)
    ind_last = int(n_samples*fraction)
    samples_filtered = samples_sorted[n_samples-ind_last:]
    samples_filtered = samples_filtered - np.min(samples_filtered)
    k_post, sigma = gpdfit(samples_filtered)
    K_list.append(k_post)


K_list = np.array(K_list)
plt.plot(1./df_list, K_list)
plt.ylabel('Degree of freedom')
plt.xlabel('1/df')
plt.savefig('tail_index_student_t.pdf')







