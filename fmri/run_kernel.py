import requests
from tqdm import tqdm
from os.path import join as oj
import tables, numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import sys
from skimage import data
import pickle as pkl
from skimage.util import img_as_float
import os
from sklearn import metrics
import h5py
from scipy.io import loadmat
from copy import deepcopy
from skimage.filters import gabor_kernel
from sklearn.linear_model import RidgeCV, ARDRegression
from sklearn.kernel_ridge import KernelRidge
sys.path.append('../lib/pymdlrs')
from src.ulnml.least_square_regression import RidgeULNML
import seaborn as sns
from scipy.io import loadmat
import numpy.linalg as npl
from scipy.optimize import minimize
import random
import sys
import scipy
from run import *
    
if __name__ == '__main__':
    np.random.seed(42) 
    random.seed(42)
    
    
    # set params
    if len(sys.argv) > 1:
        runs = [int(sys.argv[-1])]
    else:
        runs = list(range(300)) # this number determines which neuron we will pick
    print('\nruns', runs)
    
    # fit linear models
    use_sigmas = False
    out_dir = '/scratch/users/vision/data/gallant/vim_2_crcns'
    save_dir = oj(out_dir, 'mar14_kernel')
    suffix = '_feats' # _feats, '' for pixels
    norm = '_norm' # ''
    frac_train_for_cv = 0.75
    print('saving to', save_dir)
    reg_params = np.logspace(3, 6, 20).round().astype(int) # reg values to try (must match preprocess_fmri)
    
    
    print('loading data...')
    X_train = np.array(loadmat(oj(out_dir, 'mot_energy_feats_st.mat'))['S_fin'])
    X_test = np.array(loadmat(oj(out_dir, 'mot_energy_feats_sv.mat'))['S_fin'])
    X_train_kernel = load_pkl(oj(out_dir, f'mot_energy_feats_kernel_mat_ntk.pkl'))
    X_test_kernel = load_pkl(oj(out_dir, f'mot_energy_feats_kernel_test_with_train_ntk.pkl'))
    (eigenvals, eigenvecs) = load_pkl(oj(out_dir, f'eigenvals_eigenvecs_mot_energy_kernel_ntk.pkl'))

    # load the normalized responses (requires first running preprocess_fmri)
    Y_train = load_h5(oj(out_dir, 'rt_norm.h5')) # training responses: 73728 (voxels) x 7200 (timepoints)    
    Y_test = load_h5(oj(out_dir, 'rv_norm.h5') )
    sigmas = load_h5(oj(out_dir, f'out_rva_sigmas_norm.h5')) # stddev across repeats
    
    # loop over individual neurons
    for run in tqdm(runs):
        roi, i = get_roi_and_idx(run, out_dir, sigmas)
        results = {}
        os.makedirs(save_dir, exist_ok=True)
        print('fitting', roi, 'idx', i)

        # select response for neuron i
        y_train = Y_train[i]
        y_test = Y_test[i]
        if use_sigmas:
            variance = sigmas[i]**2
        else:
            variance = 1

        # count number of dims with missing time_points
        n_train = np.sum(~np.isnan(y_train))
        num_test = np.sum(~np.isnan(y_test))

        # only fit voxels with no missing vals
        if not (n_train == y_train.size and num_test == y_test.size):
            print(f'\tskipping voxel {i}!')
            continue
            
        # fit kernel ridge cv
        print('\tfitting kernel ridgecv...')
        
        mses = []
        mse_best = 1e10
        n_train_cv = int(X_train_kernel.shape[0] * frac_train_for_cv)
        X_train_cv = X_train_kernel[:n_train_cv, :n_train_cv]
        y_train_cv = y_train[:n_train_cv]
        X_val_cv = X_train_kernel[n_train_cv:, :n_train_cv]
        y_val_cv = y_train[n_train_cv:]
        for alpha in reg_params:
            m = KernelRidge(alpha=alpha, kernel='precomputed')
            m.fit(X_train_cv, y_train_cv)
            preds_val = m.predict(X_val_cv[:, :n_train_cv]) # should take kernel mat of (n_samples, n_samples_fitted)
            mse_val = metrics.mean_squared_error(y_val_cv, preds_val)
            mses.append(mse_val)
            
            # update stats for best
            if mse_val < mse_best:
                mse_best = mse_val
                alpha_best = alpha
                preds_train = m.predict(X_train_kernel[:, :n_train_cv]) # should take kernel mat of (n_samples, n_samples_fitted)
                preds = m.predict(X_test_kernel[:, :n_train_cv]) # should take kernel mat of (n_samples, n_samples_fitted)
                r2_train = metrics.r2_score(y_train, preds_train)
                mse_train = metrics.mean_squared_error(y_train, preds_train)
                mse = metrics.mean_squared_error(y_test, preds)
                r2 = metrics.r2_score(y_test, preds)
                corr = np.corrcoef(y_test, preds)[0, 1]
        print('\tKernelRidgeCV corr', corr)
        

        # fit mdl comp
        print('\tfitting kernel mdl-comp...')
        mdl_comp_opt = 1e10
        lambda_opt = None
        theta_opt = None
        r = {
            'mse_norms': [],
            'theta_norms': [],
            'eigensums': [],
            'mdl_comps': [],
            'mse_tests': [],
        }           
        for l in tqdm(reg_params):
            inv = pkl.load(open(oj(out_dir, f'pinv_mot_energy_kernel_ntk_{l}.pkl'), 'rb'))
            thetahat = X_train.T @ inv @ y_train
            mse_norm = npl.norm(y_train - X_train @ thetahat)**2 / (2 * variance)
            theta_norm = npl.norm(thetahat)**2 / (2 * variance)
            eigensum = 0.5 * np.sum(np.log(1 + eigenvals / l))
            mdl_comp = (mse_norm + theta_norm + eigensum) / n_train
            mse_test_mdl = metrics.mean_squared_error(y_test, X_test @ thetahat)            
            
            r['mse_norms'].append(mse_norm)
            r['theta_norms'].append(theta_norm)
            r['eigensums'].append(eigensum)
            r['mdl_comps'].append(mdl_comp)
            r['mse_tests'].append(mse_test_mdl)
            
            if mdl_comp < mdl_comp_opt:
                mdl_comp_opt = mdl_comp
                lambda_opt = l
                theta_opt = thetahat
                
        preds_test_mdl = X_test @ theta_opt
        mse_test_mdl = metrics.mean_squared_error(y_test, preds_test_mdl)
        
        # some misc stats
        snr = (npl.norm(y_train) ** 2 - n_train * variance) / (n_train * variance)
        y_norm = npl.norm(y_train)
        
        # save everything
        results = {
            'roi': roi,
            'model': m,
            'snr': snr,
            'lambda_best':  alpha_best,
            'n_train': n_train,
            'n_test': num_test,
            'y_norm': y_norm,
            'idx': i,
            
            # mdl stuff
            'lambda_opt': lambda_opt,
            'theta_opt': theta_opt,
            'mdl_comp_opt': mdl_comp_opt,
            'mse_test_mdl': mse_test_mdl,
            
            # cv stuff
            'frac_train_for_cv': frac_train_for_cv,
            'cv_values': mses,
            'mse_train': mse_train, 
            'r2_train': r2_train,
            'mse_test': mse,                
            'r2_test': r2,
            'corr_test': corr,
            **r,
        }
        pkl.dump(results, open(oj(save_dir, f'kernel_ridge_{i}.pkl'), 'wb'))
        print(f'\tsuccesfully finished run {run}!')