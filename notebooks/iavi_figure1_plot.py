
###  this file contains the code for plotting figure 1 of the IAVI paper  .....



#####   code to generate figure 1(a) and figure 1(b) .....
#####   python ia_figure1_plot.py  --model 5
#####   python ia_figure1_plot.py  --model 4

import numpy as np
import sys, os
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from scipy.stats import t
from itertools import product
import pystan
import pickle


from viabel.vb import (mean_field_gaussian_variational_family,
                       mean_field_t_variational_family,
                       t_variational_family,
                       make_stan_log_density)


from matplotlib import rc
rc('text', usetex = True)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 19.9}
rc('font', **font)

import  argparse
from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('/Users/akashd/Desktop/research_repos/posteriordb/posteriordb/', "posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--optimizer', default='rmsprop', type=str)
args = parser.parse_args()

from viabel.optimizers_workflow import  rmsprop_workflow_optimize1

import pickle

from experiments import black_box_klvi, psis_correction
from viabel.utils import compute_posterior_moments, data_generator_linear


modelcode = args.model


if modelcode == 5:

    regression_model_code = """data {
      int<lower=0> N;   // number of observations
      int<lower=0> D;   // number of observations
      matrix[N, D] x;   // predictor matrix
      vector[N] y;      // outcome vector
    }
    parameters {
      vector[D] beta;       // coefficients for predictors
    }

    model {
      beta ~ normal(0, 1.);
      #y ~ student_t(10000000, x * beta, 1);  // likelihood
      y ~ normal( x * beta, 0.7);  // likelihood
    }"""

    try:
        sm = pickle.load(open('linear_reg_model_400.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=regression_model_code, model_name='regression_model')
        with open('linear_reg_model_400.pkl', 'wb') as f:
            pickle.dump(sm, f)


    N = 220
    k = 80
    #SEED = 5080
    SEED=210
    alpha = 1.
    noise_sigma = 0.4
    noise_var = noise_sigma ** 2
    rho = 0.7
    num_proposal_samples = 50000
    regression_data = data_generator_linear(N, k, alpha=alpha,
                                            noise_variance=noise_var,
                                            rho=rho, seed=SEED)
    X = regression_data['X']
    Y = regression_data['Y']
    Y = Y[:, 0]
    W = regression_data['W']
    optimizer = 'rmsprop'
    data = dict(N=N, x=X, y=Y, D=k)
    fit = sm.sampling(data=data)

    prior_mean = np.zeros((k, 1))
    prior_covariance = np.eye(k)
    true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                    prior_covariance,
                                                    noise_var, X, Y[:, None])

    true_mean = true_mean.flatten()
    true_std = np.sqrt(np.diag(true_cov))

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    fr_g_var_family = t_variational_family(k, df=1000000)

    mf_g_var_family = mean_field_gaussian_variational_family(k)
    stan_log_density = make_stan_log_density(fit)
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

    approx = 'fr'
    if approx == 'mf':
        fn_density = mf_g_var_family
        init_var_param = init_var_param
        obj_and_grad = klvi_mf_objective_and_grad
    else:
        fn_density = fr_g_var_family
        init_var_param = init_fr_var_param
        obj_and_grad = klvi_fr_objective_and_grad

    if approx == 'mf' or approx == 'fr':
        if optimizer == 'rmsprop':

            try:
                with open('lr_70_fig4.pickle', 'rb') as f:
                    op_log_mf_rms = pickle.load(f)

            except:
                klvi_var_param_rms, klvi_var_param_list_rms, avg_klvi_mean_list_rms, avg_klvi_sigmas_list_rms, klvi_history_rms, _, op_log_mf_rms = \
                    rmsprop_workflow_optimize1(10000, fn_density, obj_and_grad, init_var_param, k, true_mean, true_cov, learning_rate=.018,  learning_rate_end=0.011,
                                               n_optimisers=1, stopping_rule=1, tolerance=0.02, tail_avg_iters=100)
                with open('lr_70_fig4.pickle', 'wb') as handle:
                    pickle.dump(op_log_mf_rms, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(op_log_mf_rms.keys())
            klvi_history_rms= op_log_mf_rms['nelbo']

            D_moments = op_log_mf_rms['D_moments']
            D_ia_moments = op_log_mf_rms['D_ia_moments']

            fig = plt.figure()
            fig, ax0 = plt.subplots(nrows=1, figsize=(9.1, 6.6))
            ax0.set_yscale('log')
            ax1 = ax0.twinx()
            ax1.set_ylabel('Distance D between moments')
            ax1.plot(D_moments, 'b-', label='D')
            #ax1.plot(D_ia_moments, linestyle='--', color='k')
            ax1.plot(np.arange(100, op_log_mf_rms['final_iter']), D_ia_moments, 'r--', label='D(IA)')
            #ax0.plot(klvi_history_rms, color='g', label='NELBO')
            ax0.plot(klvi_history_rms, color='k', label='NELBO')
            ax1.set_yscale('log')
            plt.savefig('lr_30_spectral_norm_constant.pdf')
            from matplotlib.pyplot import text
            vlines = [op_log_mf_rms['convergence_points'][0],  op_log_mf_rms['final_iter'], op_log_mf_rms['start_stats2']]

            h2, l2 = ax1.get_legend_handles_labels()
            h1, l1 = ax0.get_legend_handles_labels()

            ax1.legend(h2+h1, l2+l1, loc='upper center')
            ax0.axvline(x=vlines[0], color='purple', linestyle=':',label='1e-2')
            #ax0.axvline(x=vlines[1], color='red', linestyle=':', label='1e-3')
            #ax0.axvline(x=vlines[2], color='red', linestyle='-.', label='1e-4')
            ax0.axvline(x=vlines[1], color='black', linestyle=':', label='SE')
            ax0.axvline(x=vlines[2], color='orange', linestyle=':', label='R-hat')


            text_vlines= ['$\\Delta\\textrm{ELBO}<10^{-2}$', '$\\textrm{SE}<0.01$' ,'$\hat{R}< 1.1$']
            text_vlines4 = ['0.02', '0.009', '0.006', '0.002', '' ]

            #ax0.legend()
            #ax0.text(0.40, 0.5, '0.01')
            #ax0.text(0.50, 0.5, '0.001')
            #ax0.text(0.60, 0.5, '0.0001')
            #ax0.text(0.80, 0.5, 'Rhat')
            #ax0.text(0.90, 0.5, 'MCSE')
            #ax0.text(0.4, 0.1, 'khat=3.5')
            #ax0.text(0.50, 0.1, 'khat=1.2')
            #ax0.text(0.60, 0.1, 'khat=0.90')
            #ax0.text(0.90, 0.1, 'khat=0.70')
            #for i, x in enumerate(vlines):
            #    text(x+50, 5e5, text_vlines[i], rotation=90, verticalalignment='center')
            #    if i == 0:
            #        text(x - 250, 1e5, r"D:" + text_vlines4[i], rotation=0, horizontalalignment='center')
            #        text(x-250, 5e4, r"$\hat{k}:$" + text_vlines2[i], rotation=0, horizontalalignment='center')
            #        text(x-300, 1e4, r"$\hat{k}$-(IA):" + text_vlines3[i], rotation=0, horizontalalignment='center')
            #    elif i==1 or i == 2 or i==3:
            #        text(x - 200, 1e5, text_vlines4[i], rotation=0, horizontalalignment='center')
            #        text(x-120, 5e4, text_vlines2[i], rotation=0, horizontalalignment='center')
            #        text(x-120, 1e4, text_vlines3[i], rotation=0, horizontalalignment='center')


            #ax0.text(1100, 8e3, 'D(IA)', rotation=0, horizontalalignment='center')
            #ax0.text(3000, 6e4, 'D(last iteration)', rotation=0, horizontalalignment='center')

            for i, x in enumerate(vlines):
                if i==0:
                    ax0.text(x - 600, 6e6, text_vlines[i], rotation=0, verticalalignment='bottom')
                if i ==2:
                    ax0.text(x - 600, 6e6, text_vlines[i], rotation=0, verticalalignment='bottom')
                if i==1:
                    ax0.text(x - 200, 6e6, text_vlines[i], rotation=0, verticalalignment='bottom')

            ax0.set_xlabel('Iterations')
            ax0.set_ylabel('NELBO')
            #ax0.text(0, 2e3, 'NELBO')

            ax1.set_ylim((1e-1, 9))
            plt.savefig('lr_fr_new51.pdf')


            n_samples = 30000
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

            if approx == 'fr':
                cov_iters_fr_rms = fr_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
                cov_iters_fr_rms_ia = fr_g_var_family.mean_and_cov(ia_var_params)[1]
            else:
                cov_iters_fr_rms = mf_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
                cov_iters_fr_rms_ia = mf_g_var_family.mean_and_cov(ia_var_params)[1]


            print('Difference between analytical mean and HMC mean:', np.sqrt(np.sum(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))))
            print('Difference between analytical cov and HMC cov:', np.sqrt(np.sum(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))
            print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.sum(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
            print('Difference between analytical cov and HMC cov-IA:', np.sqrt(np.sum(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))


            if approx == 'mf' or approx=='fr':
                ia_difference = np.sqrt(np.sum(np.square(ia_var_params[:k].flatten() - true_mean.flatten())) + np.sum(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten())))
                print('Difference between moments:',np.sqrt(np.sum(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten())) + np.sum(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))
                print('Difference between moments(IA):',ia_difference)
                ia_difference_array = np.repeat(ia_difference, op_log_mf_rms['final_iter'])
                #ax1.plot(ia_difference_array, 'k', linestyle='--')


            else:
                print('Difference between moments:',
                      np.sum(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))) + np.sum(
                    np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))


elif modelcode == 4:
    regression_model_code = """data {
      int<lower=0> N;   // number of observations
      int<lower=0> D;   // number of observations
      matrix[N, D] x;   // predictor matrix
      vector[N] y;      // outcome vector
    }
    parameters {
      vector[D] beta;       // coefficients for predictors
    }

    model {
      beta ~ normal(0, 2.);
      #y ~ student_t(10000000, x * beta, 1);  // likelihood
      y ~ normal( x * beta, 0.5);  // likelihood
    }"""

    try:
        sm = pickle.load(open('linear_reg_model_26.pkl', 'rb'))
    except:
        sm = pystan.StanModel(model_code=regression_model_code, model_name='regression_model')
        with open('linear_reg_model_26.pkl', 'wb') as f:
            pickle.dump(sm, f)


    N = 230
    k = 30
    #SEED = 5080

    #K_list = [20, 40, 80, 100]
    SEED=210
    alpha = 1.
    noise_sigma = 0.4
    K_list = [10, 20, 30, 40, 50, 60, 80]
    K_list2 = [65, 230, 860, 1890, 3320]
    noise_var = noise_sigma ** 2
    rho = 0.7

    D_moments_last = []
    D_ia_moments_last = []

    D_moments_e2 = []
    D_ia_moments_e2 = []

    D_moments_e3 = []
    D_ia_moments_e3 = []

    D_moments_e4 = []
    D_ia_moments_e4 = []
    exclusive_kldiv_list = []
    inclusive_kldiv_list = []


    try:
        with open('lr_D_vs_D101.pickle', 'rb') as f:
            D_dict = pickle.load(f)

    except:
        for k in K_list:
            regression_data = data_generator_linear(N, k, alpha=alpha,
                                                    noise_variance=noise_var,
                                                    rho=rho, seed=SEED)
            X = regression_data['X']
            Y = regression_data['Y']
            Y = Y[:, 0]
            W = regression_data['W']

            optimizer = 'rmsprop'
            data = dict(N=N, x=X, y=Y, D=k)
            fit = sm.sampling(data=data)

            prior_mean = np.zeros((k, 1))
            prior_covariance = np.eye(k)
            true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                            prior_covariance,
                                                            noise_var, X, Y[:, None])

            true_mean = true_mean.flatten()
            true_std = np.sqrt(np.diag(true_cov))

            # print(posterior_mean)
            # print('Difference between analytical mean and HMC mean:', np.sum(np.square(posterior_mean_hmc - true_mean)))
            # print('Difference between analytical covariance and HMC covariance:', np.sum(np.square(posterior_cov_hmc - true_cov)))
            # print('Difference between analytical std dev and HMC std dev:', np.sum(np.square(posterior_std_hmc - true_std)))
            # print('Difference between analytical z-score and HMC z-score:', np.sqrt(np.mean(np.square((posterior_mean_hmc - true_mean)/true_std))))

            mf_g_var_family = mean_field_gaussian_variational_family(k)
            fr_g_var_family = t_variational_family(k, df=1000000)

            mf_g_var_family = mean_field_gaussian_variational_family(k)
            stan_log_density = make_stan_log_density(fit)
            klvi_mf_objective_and_grad = black_box_klvi(mf_g_var_family, stan_log_density, 100)
            klvi_fr_objective_and_grad = black_box_klvi(fr_g_var_family, stan_log_density, 100)
            init_mean = np.zeros(k)
            init_log_std = np.ones(k)
            init_var_param = np.concatenate([init_mean, init_log_std])
            n_iters = 5000

            klvi_mf_objective_and_grad_pd = black_box_klvi(mf_g_var_family, stan_log_density, 100)
            klvi_fr_objective_and_grad_pd = black_box_klvi(fr_g_var_family, stan_log_density, 100)
            init_mean = np.zeros(k)
            # init_mean_random = np.random.normal([k], stddev=1.0)
            init_log_std = np.ones(k)
            # init_log_std_random = tf.random.normal([k], stddev=1.)
            init_var_param = np.concatenate([init_mean, init_log_std])
            init_fr_var_param = np.concatenate([init_mean, np.ones(int(k * (k + 1) / 2))])
            approx= 'fr'

            if approx == 'mf':
                fn_density = mf_g_var_family
                init_var_param = init_var_param
                obj_and_grad = klvi_mf_objective_and_grad
            else:
                fn_density = fr_g_var_family
                init_var_param = init_fr_var_param
                obj_and_grad = klvi_fr_objective_and_grad


            if optimizer == 'rmsprop':
                klvi_var_param_rms, klvi_var_param_list_rms, avg_klvi_mean_list_rms, avg_klvi_sigmas_list_rms, klvi_history_rms, _, op_log_mf_rms1 = \
                    rmsprop_workflow_optimize1(11000, fn_density, obj_and_grad, init_var_param, k, true_mean, true_cov, learning_rate=.018, learning_rate_end=0.010, n_optimisers=1, stopping_rule=1, tolerance=0.02, plotting=False)

                #D_moments = np.sqrt(op_log_mf_rms1['D_moments'])
                #D_ia_moments = np.sqrt(op_log_mf_rms1['D_ia_moments'])
                D_moments = op_log_mf_rms1['D_moments']
                D_ia_moments = op_log_mf_rms1['D_ia_moments']

                D_moments_last.append(D_moments[-1])
                D_ia_moments_last.append(D_ia_moments[-1])

                D_moments_e2.append(D_moments[op_log_mf_rms1['convergence_points'][0]])
                D_ia_moments_e2.append(D_ia_moments[op_log_mf_rms1['convergence_points'][0]])
                #D_moments_e3.append(D_moments[op_log_mf_rms1['convergence_points'][1]])
                #D_ia_moments_e3.append(D_moments[op_log_mf_rms1['convergence_points'][1]])

                #D_moments_e4.append(D_moments[op_log_mf_rms1['convergence_points'][2]])
                #D_ia_moments_e4.append(D_moments[op_log_mf_rms1['convergence_points'][2]])
                n_samples = 40000
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
                true_mean, true_cov = compute_posterior_moments(prior_mean,
                                                                prior_covariance,
                                                                noise_var, X, Y[:, None])


                if approx == 'fr':
                    cov_iters_fr_rms = fr_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
                    cov_iters_fr_rms_ia = fr_g_var_family.mean_and_cov(ia_var_params)[1]
                else:
                    cov_iters_fr_rms = mf_g_var_family.mean_and_cov(klvi_var_param_rms)[1]
                    cov_iters_fr_rms_ia = mf_g_var_family.mean_and_cov(ia_var_params)[1]


                inclusive_kldiv = kldiv(ia_var_params[:k], cov_iters_fr_rms_ia, true_mean, true_cov)
                exclusive_kldiv = kldiv(true_mean, true_cov, ia_var_params[:k], cov_iters_fr_rms_ia)
                #print('Difference between analytical mean and HMC mean:', np.sqrt(np.mean(np.square(klvi_var_param_rms[:k].flatten() - true_mean.flatten()))))
                #print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms.flatten() - true_cov.flatten()))))

                #print('Difference between analytical mean and HMC mean-IA:', np.sqrt(np.mean(np.square(ia_var_params[:k].flatten() - true_mean.flatten()))))
                #print('Difference between analytical cov and HMC cov:', np.sqrt(np.mean(np.square(cov_iters_fr_rms_ia.flatten() - true_cov.flatten()))))

        D_dict = {}
        D_dict['D_mcse_last'] = D_moments_last
        D_dict['D_mcse_ia_last'] = D_ia_moments_last
        D_dict['D_elbo_last'] = D_moments_e2
        D_dict['D_elbo_ia_last'] = D_ia_moments_e2

        with open('lr_D_vs_D101.pickle', 'wb') as handle:
            pickle.dump(D_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    fig= plt.figure()
    fig, ax0 = plt.subplots(nrows=1, figsize=(9.1, 6.4))
    fig, ax0 = plt.subplots(nrows=1, figsize=(9.1, 6.4))

    ax0.plot(K_list2,  D_dict['D_elbo_last'], 'b-', label='$\Delta\\textrm{ELBO} < 0.01$(last iterate)')
    ax0.plot(K_list2,  D_dict['D_elbo_ia_last'], 'b--', label='$\Delta\\textrm{ELBO} < 0.01$(iterate average)')
    ax0.plot(K_list2,   D_dict['D_mcse_last'], 'r', label='$\\textrm{MCSE} < 0.01$(last iterate)')
    ax0.plot(K_list2, D_dict['D_mcse_ia_last'], 'r--', label='$\\textrm{MCSE} < 0.01$(iterate average)')
    ax0.set_yscale('log')
    ax0.set_xlabel('Dimensions of variational parameter(K)')
    ax0.set_ylabel('Distance D between moments')
    box = ax0.get_position()
    ax0.set_position([box.x0, box.y0, box.width, box.height*0.89])
    #ax0.legend(loc='upper center')
    ax0.legend(loc='lower left', bbox_to_anchor=(-0.15, 1.02, 1., .102), ncol=2, borderaxespad=0.)
    #plt.plot(K_list, D_moments_e3)
    #plt.plot(K_list, D_ia_moments_e3)
    #plt.plot(K_list, D_moments_e4)
    #plt.plot(K_list, D_ia_moments_e4)
    plt.savefig('lin_reg_DvsK101.pdf')


