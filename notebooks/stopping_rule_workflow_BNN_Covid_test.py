


import sys, os
import pickle
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from scipy.stats import t
from itertools import product
import pystan

from viabel.vb import (mean_field_gaussian_variational_family,
                       mean_field_t_variational_family,
                       full_rank_gaussian_variational_family,
                       t_variational_family,
                       make_stan_log_density,
                       adagrad_optimize)


import  argparse
from posteriordb import PosteriorDatabase
import os

# please give path to your posteriordb installation here ......
pdb_path = os.path.join('/Users/akashd/Desktop/research_repos/posteriordb/posteriordb/', "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()



parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--optimizer', default='rmsprop', type=str)
args = parser.parse_args()

modelcode = args.model

from experiments import black_box_klvi, psis_correction
from viabel.functions import compute_posterior_moments
from data_generator import (data_generator_linear)

from viabel.vb import  rmsprop_IA_optimize_with_rhat, adam_IA_optimize_with_rhat
#from viabel.optimizers_avg_stopping_rule import  adam_IA_optimize_stop, adagrad_ia_optimize_stop, rmsprop_IA_optimize_stop
from viabel.optimizers_workflow import adagrad_workflow_optimize, rmsprop_workflow_optimize, adam_workflow_optimize


from viabel.data_process  import  Concrete
#approx= 'mf'

if modelcode ==1:
    model_str = "ecdc0501-covid19imperial_v3"

elif modelcode == 2:
    model_str = "ecdc0401-covid19imperial_v3"

elif modelcode == 3:
    model_str = "ecdc0401-covid19imperial_v2"

elif modelcode == 5:
    model_str= 'concrete'

posterior = my_pdb.posterior(model_str)
modelObject = posterior.model
data= posterior.data
code_string = modelObject.code('stan')
#text_file = open("stan_models/stan-covid19imperial_v2.stan", "w")


optimizer = 'rmsprop'

# Concrete just as a running example.
if modelcode == 5:
    CData = Concrete()
    X1, Y1 = CData.get_normalised_data()

    print(X1.shape)
    print(Y1.shape)
    # exit()
    N_train, k = X1.shape

    linear_reg_code = """
    data{
        int<lower=0> N;
        int<lower=0> K;
        matrix[N,K] X;
        vector[N] y;
        real<lower=0> sigma;
    }
    parameters{
    vector[K] w;
    #real<lower=0> sigma;
    }
    model{
    w ~ normal(0,1);
    y ~ normal(X*w , sigma);
    }
    generated quantities{
    real log_density;
    #log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1) + gamma_lpdf(sigma|0.5, 0.5) + log(sigma);
    #log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1)+ gamma_lpdf(sigma|0.5, 0.5) + log(sigma);
    log_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1);
    }
    """

    model_data = {'N': N_train,
                  'K': k,
                  'y': Y1[:, 0],
                  'X': X1,
                  'sigma':0.2

                  }

    try:
        sm = pickle.load(open('linear_reg_chains_concrete5.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=linear_reg_code)
        with open('linear_reg_chains_concrete5.pkl', 'wb') as f:
            pickle.dump(sm, f)

    fit_hmc = sm.sampling(data=model_data, iter=2400)

    la = fit_hmc.extract(permuted=True)
    hmc_w = la['w']
    #hmc_residual = la['sigma']
    # stan_sigma = la['sigma']
    # stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta
    params_hmc = hmc_w
    # params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
    true_mean = np.mean(params_hmc, axis=0)
    true_cov = np.cov(params_hmc.T)
    params_hmc_sq = np.mean(params_hmc ** 2, axis=0)
    params_hmc_sigmas = np.std(params_hmc, axis=0)

    #params_hmc_residual_mean = np.mean(hmc_residual, axis=0)
    #params_hmc_residual_sq = np.mean(hmc_residual ** 2, axis=0)
    #params_hmc_residual_sigmas = np.std(hmc_residual, axis=0)
    print('##### HMC Mean####')

    num_proposal_samples = 50000

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    fr_g_var_family = t_variational_family(k, df=1000000)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    stan_log_density = make_stan_log_density(fit_hmc)
    klvi_mf_objective_and_grad = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    init_log_std = np.ones(k)
    init_var_param = np.concatenate([init_mean, init_log_std])
    n_iters = 9000

    klvi_mf_objective_and_grad_pd = black_box_klvi(mf_g_var_family, stan_log_density, 100)
    klvi_fr_objective_and_grad_pd = black_box_klvi(fr_g_var_family, stan_log_density, 100)
    init_mean = np.zeros(k)
    # init_mean_random = np.random.normal([k], stddev=1.0)
    init_log_std = np.ones(k)
    # init_log_std_random = tf.random.normal([k], stddev=1.)
    init_var_param = np.concatenate([init_mean, init_log_std])
    init_fr_var_param = np.concatenate([init_mean, np.ones(int(k * (k + 1) / 2))])
    optimizer = 'rmsprop'

    fn_density = fr_g_var_family
    init_var_param = init_fr_var_param
    obj_and_grad = klvi_fr_objective_and_grad

    if optimizer == 'rmsprop':
        klvi_var_param_rms, klvi_var_param_list_rms, avg_klvi_mean_list_rms, avg_klvi_sigmas_list_rms, klvi_history_rms, _, op_log_mf_rms = \
            rmsprop_workflow_optimize(11000, obj_and_grad, init_var_param, k, learning_rate=.006, n_optimisers=1, stopping_rule=1, tolerance=0.005)

        n_samples = 20000
        ia_var_params=  np.concatenate((avg_klvi_mean_list_rms[0], avg_klvi_sigmas_list_rms[0]), axis=0)
        print(ia_var_params)

        samples, smoothed_log_weights, khat = psis_correction(stan_log_density, fn_density,
                                                             klvi_var_param_list_rms[0,-1,:], n_samples)

        samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(stan_log_density, fn_density,
                                                             ia_var_params, n_samples)

        print(true_mean)
        print(klvi_var_param_rms[:k])
        print('khat:', khat)
        print('khat-ia:', khat_ia)

        cov_iters_fr_rms = fr_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
        cov_iters_fr_rms_ia = fr_g_var_family.mean_and_cov(ia_var_params)[1]
        print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

        print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
        print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))

# code for covid19 model starts from here ...
elif modelcode == 2:
    print('lol')
    try:
        # save the code string ....
        sm = pickle.load(open('stan_pkl/covid19_01_v3.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_2')
        with open('stan_pkl/covid19_01_v3.pkl', 'wb') as f:
            pickle.dump(sm, f)


    try:
        # save the posterior model .....
        model_fit = pickle.load(open('stan_pkl/covid19_v3_posterior_samples.pkl', 'rb'))
    except:
        model_fit = sm.sampling(data=data.values(), iter=800,
                                                 control=dict(adapt_delta=.96), chains=1)
        with open('stan_pkl/covid19_v3_posterior_samples.pkl', 'wb') as f:
            pickle.dump(model_fit, f)

    #sm = pystan.StanModel(model_code=code_string, model_name='covid19_model_v2')
    #model_fit = sm.sampling(data=data.values(), iter=600, chains=1)
    K = len(model_fit.constrained_param_names())
    print(K)
    param_names =  model_fit.flatnames
    # construct matrix of samples (both original and transformed) from non-centered model
    samples_posterior = model_fit.to_dataframe(pars=model_fit.flatnames)
    #samples_posterior['log_sigma'] = np.log(samples_posterior['sigma'])
    samples_posterior = samples_posterior.loc[:,param_names].values.T

    true_mean = np.mean(samples_posterior, axis=1)
    true_cov = np.cov(samples_posterior)
    true_sigma = np.sqrt(np.diag(true_cov))
    covid19_log_density = make_stan_log_density(model_fit)

    true_mean_pmz = true_mean[:K]
    true_sigma_pmz = true_sigma[:K]

    mf_gaussian = mean_field_gaussian_variational_family(K)
    fr_gaussian = t_variational_family(K, df=10000000)
    init_param_fr = np.concatenate([np.zeros(K), np.ones(int(K*(K+1)/2))])
    #init_param_mf = np.concatenate([np.zeros(K), np.ones(K)])
    init_param_mf = np.concatenate([true_mean_pmz, np.log(true_sigma_pmz)])

    klvi_fr_objective_and_grad = black_box_klvi(fr_gaussian, covid19_log_density, 100)
    klvi_mf_objective_and_grad = black_box_klvi(mf_gaussian, covid19_log_density, 100)

    approx = 'fr'
    # define the relevant functions based on the approximate density.
    if approx == 'mf':
        fn_density = mf_gaussian
        init_var_param = init_param_mf
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_gaussian
        init_var_param = init_param_fr
        obj_and_grad = klvi_fr_objective_and_grad

    print('########### HMC Mean #################')
    print(true_mean)
    print(true_sigma)
    n_samples=100000

    a, b, c, d, e = \
        adagrad_workflow_optimize(10000, obj_and_grad, init_var_param,
                                  K, learning_rate=.0004, n_optimizers=1, tolerance=0.05, stopping_rule=1)
    samples, smoothed_log_weights, khat = psis_correction(covid19_log_density, fn_density,
                                                          b[-1], n_samples)
    samples_ia, smoothed_log_weights_ia, khat_ia = psis_correction(covid19_log_density, fn_density,
                                                                   a, n_samples)
    print(true_mean)
    print(b[-1][:K])
    print('khat:', khat)
    print('khat ia:', khat_ia)
    cov_iters_fr_rms = fn_density.mean_and_cov(b[-1])[1]
    cov_iters_fr_rms_ia1 = fn_density.mean_and_cov(a)[1]
    print('Difference between analytical mean and HMC mean:',
          np.sqrt(np.mean(np.square(b[-1][:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

    print('Difference between analytical mean and HMC mean-IA:',
          np.sqrt(np.mean(np.square(a[:K].flatten() - true_mean.flatten()))))
    print('Difference between analytical cov and HMC cov-IA:',
          np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia1.flatten() - true_cov.flatten()))))