import requests
from tqdm import tqdm
from os.path import join as oj
import tables, numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
import pickle as pkl
from skimage.util import img_as_float
import os
from sklearn import metrics
import h5py
from scipy.io import loadmat
from copy import deepcopy
from skimage.filters import gabor_kernel
import gabor_feats
from sklearn.linear_model import RidgeCV
import seaborn as sns
from scipy.io import loadmat
import numpy.linalg as npl
from scipy.optimize import minimize
import random
import sys

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

def save_pkl(d, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pkl.dump(d, f)
    
def get_roi_and_idx(run):
    # select roi + i (which is the roi_idx)
    rois = ['v1lh', 'v2lh', 'v4lh', 'v1rh', 'v2rh', 'v4rh']
    roi = rois[run % len(rois)]

    f = tables.open_file(oj(out_dir, 'VoxelResponses_subject1.mat'), 'r')
    roi_idxs_all = f.get_node(f'/roi/{roi}')[:].flatten().nonzero()[0] # structure containing volume matrices (64x64x18) with indices corresponding to each roi in each hemisphere
    roi_idxs = np.array([roi_idx for roi_idx in roi_idxs_all if ~np.isnan(sigmas[roi_idx])])

    i = roi_idxs[run // len(rois)] # i is the roi idx
    return roi, i  
    
if __name__ == '__main__':
    np.random.seed(42) 
    random.seed(42)
    
    
    # set params
    if len(sys.argv) > 1:
        runs = [int(sys.argv[-1])]
    else:
        runs = list(range(300)) # this number determines which neuron we will pick
    print(runs)
    
    # fit linear models
    out_dir = '/scratch/users/vision/data/gallant/vim_2_crcns'
    save_dir = oj(out_dir, 'test')
    suffix = '_feats' # _feats, '' for pixels
    norm = '_norm' # ''
    

    print('loading data...')
    '''
    feats_name = oj(out_dir, f'out_st{suffix}{norm}.h5')
    feats_test_name = oj(out_dir, f'out_sv{suffix}{norm}.h5')
    X = np.array(h5py.File(feats_name, 'r')['data'])
    X = X.reshape(X.shape[0], -1)
    print('shape, Y.shape', Y.shape)
    X_test = np.array(h5py.File(feats_test_name, 'r')['data'])
    X_test = X_test.reshape(X_test.shape[0], -1)
    '''
    X = np.array(loadmat(oj(out_dir, 'mot_energy_feats_st.mat'))['S_fin'])
    X_test = np.array(loadmat(oj(out_dir, 'mot_energy_feats_sv.mat'))['S_fin'])
    
    '''
    resps_name = oj(out_dir, 'VoxelResponses_subject1.mat')
    Y = np.array(tables.open_file(resps_name).get_node(f'/rt')[:]) # training responses: 73728 (voxels) x 7200 (timepoints)
    Y_test = np.array(tables.open_file(resps_name).get_node(f'/rv')[:]) 
    '''
    Y = load_h5(oj(out_dir, 'rt_norm.h5')) # training responses: 73728 (voxels) x 7200 (timepoints)    
    Y_test = load_h5(oj(out_dir, 'rv_norm.h5') )
    sigmas = load_h5(oj(out_dir, f'out_rva_sigmas.h5'))
    (U, alphas, _) = pkl.load(open(oj(out_dir, f'decomp_mot_energy.pkl'), 'rb'))
    
    # loop over individual neurons
    for run in runs:
        roi, i = get_roi_and_idx(run)
        results = {}
        os.makedirs(save_dir, exist_ok=True)
        print('fitting', roi, 'idx', i)

        # load stuff
        y = Y[i]
        y_test = Y_test[i]
        w = U.T @ y
        sigma = sigmas[i]
        var = sigma**2

        # ignore voxels w/ missing vals
        idxs_cv = ~np.isnan(y)
        idxs_test = ~np.isnan(y_test)
        n = np.sum(idxs_cv)
        num_test = np.sum(idxs_test)
        d = X.shape[1]
        d_n_min = min(n, d)

        if n == y.size and num_test == y_test.size: 
            m = RidgeCV(alphas=[1e3, 2.5e3, 5e3, 7.5e3, 1e4, 2.5e4, 5e4, 
                                7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6])
            m.fit(X, y)
            preds_train = m.predict(X)
            preds = m.predict(X_test)
            mse_train = metrics.mean_squared_error(y, preds_train)
            r2_train = metrics.r2_score(y, preds_train)
            mse = metrics.mean_squared_error(y_test, preds)
            r2 = metrics.r2_score(y_test, preds)
            corr = np.corrcoef(y_test, preds)[0, 1]
    #                 print('w', npl.norm(w), 'y', npl.norm(y), 'var', var)
            term1 = 0.5 * (npl.norm(y) ** 2 - npl.norm(w) ** 2) / var
            term2 = 0.5 * np.sum([np.log(1 + w[i]**2 / var) for i in range(d_n_min)])
            complexity1 = term1 + term2
    #                 print('term1', term1, 'term2', term2) #, 'alpha', m.alpha_)

            idxs = np.abs(w) > sigma
            term3 = 0.5 * np.sum([np.log(1 + w[i]**2 / var) for i in np.arange(n)[idxs]])
            term4 = 0.5 * np.sum([w[i]**2 / var for i in np.arange(n)[~idxs]])
            complexity2 = term1 + term3 + term4
            
            snr = (npl.norm(y) ** 2 - n * var) / (n * var)
            y_norm = npl.norm(y)
            

            results = {
                'roi': roi,
                'model': m,
                'term1': term1,
                'term2': term2,
                'term3': term3,
                'term4': term4,
                'complexity1': complexity1 / n,
                'complexity2': complexity2 / n,
                'snr': snr,
                'lambda_best':  m.alpha_,
                'n_train': n,
                'n_test': num_test,
                'd': d,
                'y_norm': y_norm,
                'mse_train': mse_train,
                'r2_train': r2_train,
                'mse_test': mse,                
                'r2_test': r2,
                'corr_test': corr,
                'idx': i
            }
            pkl.dump(results, open(oj(save_dir, f'ridge_{i}.pkl'), 'wb'))