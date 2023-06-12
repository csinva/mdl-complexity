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
from sklearn.linear_model import RidgeCV, ARDRegression, LinearRegression, Ridge
sys.path.append('../lib/pymdlrs')
from src.ulnml.least_square_regression import RidgeULNML
import seaborn as sns
from scipy.io import loadmat
import numpy.linalg as npl
from scipy.optimize import minimize
import random

class RidgeBICRegressor():
    def __init__(self):
        self.alphas = np.logspace(-1, 3, 20).round()

    def fit(self, X, y):
        n, d = X.shape
        bic_scores = []
        models = []
        
        ols = LinearRegression()
        denom = np.std(y - ols.fit(X, y).predict(X)) / (n - d)
        
        for alpha in tqdm(self.alphas):
            model = Ridge(alpha=alpha, fit_intercept=False, normalize=False)
            model.fit(X, y)
            models.append(model)
            
            # key lines
            n_feats = np.trace(X @ npl.inv(X.T @ X + alpha * np.eye(d)) @ X.T) 
            rss = np.sum((model.predict(X) - y) ** 2) / denom
            bic = n * np.log(rss / n) + n_feats * np.log(n)
            bic_scores.append(bic)
    
        best_model_index = np.argmin(bic_scores)
        self.alpha_ = self.alphas[best_model_index]
        self.model_ = models[best_model_index]
    
    def predict(self, X):
        return self.model_.predict(X)

def save_h5(data, fname):
    if os.path.exists(fname):
        os.remove(fname)
    f = h5py.File(fname, 'w')
    f['data'] = data
    f.close()    

def load_h5(fname):
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    f.close()
    return data

def load_pkl(fname):
    return pkl.load(open(fname, "rb" ))

def save_pkl(d, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pkl.dump(d, f)
    
def get_roi_and_idx(run, out_dir, sigmas):
    # select roi + i (which is the roi_idx)
    rois = ['v1lh', 'v2lh', 'v4lh', 'v1rh', 'v2rh', 'v4rh']
    roi = rois[run % len(rois)]

    f = tables.open_file(oj(out_dir, 'VoxelResponses_subject1.mat'), 'r')
    roi_idxs_all = f.get_node(f'/roi/{roi}')[:].flatten().nonzero()[0] # structure containing volume matrices (64x64x18) with indices corresponding to each roi in each hemisphere
    roi_idxs = np.array([roi_idx for roi_idx in roi_idxs_all
                         if ~np.isnan(sigmas[roi_idx])])

    i = roi_idxs[run // len(rois)] # i is the roi idx
    return roi, i  

    
if __name__ == '__main__':
    np.random.seed(42) 
    random.seed(42)
    
    
    # set params
    if len(sys.argv) > 1:
        runs = [int(sys.argv[-1])]
    else:
        runs = list(range(100)) # this number determines which neuron we will pick
    print('\nruns', runs)
    
    # fit linear models
    use_sigmas = False
    use_small = False
    out_dir = '/scratch/users/vision/data/gallant/vim_2_crcns'
    # save_dir = oj(out_dir, 'dec14_baselines_ard')
    save_dir = oj(out_dir, 'jun7_baselines_bic')
    suffix = '_feats' # _feats, '' for pixels
    norm = '_norm' # ''
    reg_params = np.logspace(3, 6, 20).round().astype(int) # reg values to try (must match preprocess_fmri)
    print('saving to', save_dir)
    

    print('loading data...')
    '''
    # load the raw pixel data
    feats_name = oj(out_dir, f'out_st{suffix}{norm}.h5')
    feats_test_name = oj(out_dir, f'out_sv{suffix}{norm}.h5')
    X_train = np.array(h5py.File(feats_name, 'r')['data'])
    X_train = X_train.reshape(X_train.shape[0], -1)
    print('shape, Y.shape', Y.shape)
    X_test = np.array(h5py.File(feats_test_name, 'r')['data'])
    X_test = X_test.reshape(X_test.shape[0], -1)
    '''
    # load the motion energy features
    if use_small:
        X_train = load_h5(oj(out_dir, f'mot_energy_feats_st_small.h5'))
    else:
        X_train = np.array(loadmat(oj(out_dir, 'mot_energy_feats_st.mat'))['S_fin'])
    X_test = np.array(loadmat(oj(out_dir, 'mot_energy_feats_sv.mat'))['S_fin'])
    if use_small:
#         (U, alphas, _) = pkl.load(open(oj(out_dir, f'decomp_mot_energy_small.pkl'), 'rb'))
        (eigenvals, eigenvecs) = pkl.load(open(oj(out_dir, f'eigenvals_eigenvecs_mot_energy_small.pkl'), 'rb'))        
        Y_train = Y_train[:, :720]
    else:
#         (U, alphas, _) = pkl.load(open(oj(out_dir, f'decomp_mot_energy.pkl'), 'rb'))
        (eigenvals, eigenvecs) = pkl.load(open(oj(out_dir, f'eigenvals_eigenvecs_mot_energy.pkl'), 'rb'))
    
    
    '''
    # load the raw responses
    resps_name = oj(out_dir, 'VoxelResponses_subject1.mat')
    Y_train = np.array(tables.open_file(resps_name).get_node(f'/rt')[:]) # training responses: 73728 (voxels) x 7200 (timepoints)
    Y_test = np.array(tables.open_file(resps_name).get_node(f'/rv')[:]) 
    '''
    # load the normalized responses (requires first running preprocess_fmri)
    Y_train = load_h5(oj(out_dir, 'rt_norm.h5')) # training responses: 73728 (voxels) x 7200 (timepoints)    
    Y_test = load_h5(oj(out_dir, 'rv_norm.h5') )
    sigmas = load_h5(oj(out_dir, f'out_rva_sigmas_norm.h5')) # stddev across repeats

    
    
    # loop over individual neurons
    for run in runs:
        roi, i = get_roi_and_idx(run, out_dir, sigmas)
        results = {}
        os.makedirs(save_dir, exist_ok=True)
        print('fitting', roi, 'idx', i)


        # check for cached file
        cached_fname = oj(save_dir, f'ridge_{i}.pkl')
        if os.path.exists(cached_fname):
            print('skipping', i)
            exit(0)

        # select response for neuron i
        y_train = Y_train[i]
        y_test = Y_test[i]
        # w = U.T @ y_train
        if use_sigmas:
            variance = sigmas[i]**2
        else:
            variance = 1

        # count number of dims with missing time_points
        n_train = np.sum(~np.isnan(y_train))
        num_test = np.sum(~np.isnan(y_test))
        d = X_train.shape[1]

        # only fit voxels with no missing vals
        if not (n_train == y_train.size and num_test == y_test.size):
            print('\tskipping this voxel!')
            continue
        
        # fit ard + mdl-rs
        baselines = {}
        for model_type, model_name in zip([RidgeBICRegressor], ['bic']):
        # for model_type, model_name in zip([ARDRegression], ['ard']):
#         for model_type, model_name in zip([ARDRegression, RidgeULNML], ['ard', 'mdl-rs']):
            print('\tfitting', model_name)
            model = model_type()
            model.fit(X_train, y_train)
            preds_train = model.predict(X_train)
            preds = model.predict(X_test)
            baselines[f'{model_name}_mse_train'] = metrics.mean_squared_error(y_train, preds_train)
            baselines[f'{model_name}_r2_train'] = metrics.r2_score(y_train, preds_train)
            baselines[f'{model_name}_mse'] = metrics.mean_squared_error(y_test, preds)
            baselines[f'{model_name}_r2'] = metrics.r2_score(y_test, preds)
            baselines[f'{model_name}_corr'] = np.corrcoef(y_test, preds)[0, 1]
            
        """
        # fit ridge cv
        print('\tfitting ridgecv...')
        m = RidgeCV(alphas=reg_params, store_cv_values=True)
        m.fit(X_train, y_train)
        preds_train = m.predict(X_train)
        preds = m.predict(X_test)
        mse_train = metrics.mean_squared_error(y_train, preds_train)
        r2_train = metrics.r2_score(y_train, preds_train)
        mse = metrics.mean_squared_error(y_test, preds)
        r2 = metrics.r2_score(y_test, preds)
        corr = np.corrcoef(y_test, preds)[0, 1]
        print('\tRidgeCV corr', corr)
        

        # fit mdl comp
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
            if use_small:
                inv = pkl.load(open(oj(out_dir, f'invs/pinv_mot_energy_st_{l}_small.pkl'), 'rb'))
            else:
                inv = pkl.load(open(oj(out_dir, f'pinv_mot_energy_st_{l}.pkl'), 'rb'))
            thetahat = inv @ X_train.T @ y_train
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
            'lambda_best':  m.alpha_,
            'n_train': n_train,
            'n_test': num_test,
            'd': d,
            'y_norm': y_norm,
            'idx': i,
            
            # mdl stuff
            'lambda_opt': lambda_opt,
            'theta_opt': theta_opt,
            'mdl_comp_opt': mdl_comp_opt,
            'mse_test_mdl': mse_test_mdl,
            
            # cv stuff
            'cv_values': m.cv_values_,
            'mse_train': mse_train, 
            'r2_train': r2_train,
            'mse_test': mse,                
            'r2_test': r2,
            'corr_test': corr,
            **r,
            **baselines,
        }
        """
        results = {
            'roi': roi,
            # 'model': m,
            # 'snr': snr,
            # 'lambda_best':  m.alpha_,
            'n_train': n_train,
            'n_test': num_test,
            'd': d,
            # 'y_norm': y_norm,
            'idx': i,
            **baselines,
        }
        pkl.dump(results, open(oj(save_dir, f'ridge_{i}.pkl'), 'wb'))
        print(f'\tsuccesfully finished run {run}!')