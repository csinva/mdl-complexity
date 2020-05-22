from torch.autograd import Variable
import torch
import torch.autograd
import torch.nn.functional as F
import random
import numpy as np
# from params import p
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import pickle as pkl
from os.path import join as oj
import numpy.random as npr
import numpy.linalg as npl
from copy import deepcopy
import pandas as pd
import seaborn as sns
from scipy.stats import random_correlation
from scipy.stats import t

# datasets with at least 15 features
REGRESSION_DSETS_LARGE_NAMES_RECOGNIZABLE = ['1028_SWD', '1201_BNG_breastTumor', '192_vineyard', '201_pol', '210_cloud', '215_2dplanes', '1029_LEV', '218_house_8L', '225_puma8NH', '228_elusage', '229_pwLinear', '230_machine_cpu', '294_satellite_image', '344_mv', '4544_GeographicalOriginalofMusic', '1030_ERA', '519_vinnie', '522_pm10', '523_analcatdata_neavote', '527_analcatdata_election2000', '529_pollen', '537_houses', '542_pollution', '1096_FacultySalaries', '1191_BNG_pbc', '1193_BNG_lowbwt', '1196_BNG_pharynx', '1199_BNG_echoMonths']


# get means and covariances
def get_means_and_cov(num_vars, iid='clustered', cov_param=None):
    means = np.zeros(num_vars)
    inv_sum = num_vars
    if iid == 'clustered':
        eigs = []
        while len(eigs) < num_vars - 1:
            if inv_sum <= 1e-2:
                eig = 0
            else:
                eig = np.random.uniform(0, inv_sum)
            eigs.append(eig)
            inv_sum -= eig

        eigs.append(num_vars - np.sum(eigs))
        covs = random_correlation.rvs(eigs)
    elif iid == 'spike':
        covs = random_correlation.rvs(np.ones(num_vars)) # basically identity with some noise
        covs = covs + 0.5 * np.ones(covs.shape)
    elif iid == 'decay':
        eigs = np.array([1/((i + 1) ** cov_param) for i in range(num_vars)])
        eigs = eigs * num_vars / eigs.sum()
        covs = random_correlation.rvs(eigs)
    elif iid == 'mult_decay':
        eigs = np.array([cov_param ** (i) for i in range(num_vars)])
        eigs = eigs * num_vars / eigs.sum()
        covs = random_correlation.rvs(eigs)
    elif iid == 'lin_decay':
        eigs = np.array([1/(cov_param ** i) for i in range(num_vars)])
        eigs = eigs * num_vars / eigs.sum()
        covs = random_correlation.rvs(eigs)
    return means, covs


def get_X(n, p, iid, means=None, covs=None, cov_param=None):
    if iid == 'iid':
        X = np.random.randn(n, p)
    elif iid in ['clustered', 'spike', 'decay', 'mult_decay', 'lin_decay']:
        means, covs = get_means_and_cov(p, iid, cov_param)
        X = np.random.multivariate_normal(means, covs, (n,))
    else:
        print(iid, ' data not supported!')
    return X, means, covs
    

def get_Y(X, beta, noise_std, noise_distr):
    if noise_distr == 'gaussian':
        return X @ beta + noise_std * np.random.randn(X.shape[0])
    elif noise_distr == 't': # student's t w/ 3 degrees of freedom
        return X @ beta + noise_std * t.rvs(df=3, size=X.shape[0])
    elif noise_distr == 'gaussian_scale_var': # want variance of noise to scale with squared norm of x
        return X @ beta + noise_std * np.multiply(np.random.randn(X.shape[0]), np.linalg.norm(X, axis=1))
    elif noise_distr == 'thresh':
        return (X > 0).astype(np.float32) @ beta + noise_std * np.random.randn(X.shape[0])
    

def get_data_train_test(n_train=10, n_test=100, p=10000, noise_std=0.1, noise_distr='gaussian', iid='iid',
                        beta_type='one_hot', beta_norm=1, seed_for_training_data=None, cov_param=None):

    '''Get data for simulations - test should always be the same given all the parameters (except seed_for_training_data)
    Warning - this sets a random seed!
    '''
    
    np.random.seed(seed=703858704)
    
    # get beta
    if beta_type == 'one_hot':
        beta = np.zeros(p)
        beta[0] = 1
    elif beta_type == 'gaussian':
        beta = np.random.randn(p)
    beta = beta / np.linalg.norm(beta) * beta_norm
        
    
    # data
    X_test, means, covs = get_X(n_test, p, iid, cov_param=cov_param)
    y_test = get_Y(X_test, beta, noise_std, noise_distr)
    
    # re-seed before getting betastar
    if not seed_for_training_data is None:
        np.random.seed(seed=seed_for_training_data)
    
    X_train, _, _ = get_X(n_train, p, iid, means, covs, cov_param)
    y_train = get_Y(X_train, beta, noise_std, noise_distr)
    
    return X_train, y_train, X_test, y_test, beta