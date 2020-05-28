import numpy as np
import traceback
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn import metrics
import data
from tqdm import tqdm
import pickle as pkl
from copy import deepcopy
import time
import random
from os.path import join as oj
import os
import sys
from params import S as S_save
import pmlb
from scipy.optimize import minimize
import numpy.linalg as npl

def seed(s):
    '''set random seed        
    '''
    np.random.seed(s) 
    random.seed(s)
    
def save(out_name, p, s):
    if not os.path.exists(p.out_dir):  
        os.makedirs(p.out_dir)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict()}    
    pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))
    
def fit(p):
    out_name = p._str(p) # generate random fname str before saving
    seed(p.seed)
    s = S_save(p)
    
    
    #################################################################### DATA ##############################################################
    
    # testing data should always be generated with the same seed
    if p.dset == 'gaussian':
        p.n_train = int(p.n_train_over_num_features * p.num_features)

        # warning - this reseeds!
        X_train, y_train, X_test, y_test, s.betastar = \
            data.get_data_train_test(n_train=p.n_train, n_test=p.n_test, p=p.num_features, 
                                noise_std=p.noise_std, noise_distr=p.noise_distr, iid=p.iid, # parameters to be determined
                                beta_type=p.beta_type, beta_norm=p.beta_norm, 
                                seed_for_training_data=p.seed, cov_param=p.cov_param)
    elif p.dset == 'pmlb':
        s.dset_name = data.REGRESSION_DSETS_LARGE_NAMES_RECOGNIZABLE[p.dset_num]
        seed(703858704)
        X, y = pmlb.fetch_data(s.dset_name, return_X_y=True)
        # normalize the data
        X = (X - np.mean(X, axis=1).reshape(-1, 1)) / np.std(X, axis=1).reshape(-1, 1)
        y = (y - np.mean(y)) / np.std(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y) # get test set
        seed(p.seed)
        X_train, y_train = shuffle(X_train, y_train)
        p.num_features = X_train.shape[1]
        p.n_train = int(p.n_train_over_num_features * p.num_features)
        '''
        while p.n_train <= X_train.shape[0]:
            X_train = np.vstack((X_train, 
                                 1e-3 * np.random.randn(X_train.shape[0], X_train.shape[1])))
            y_train = np.vstack((y_train, y_train))
        '''
        if p.n_train > X_train.shape[0]:
            print('this value of n too large')
            exit(0)
        elif p.n_train <= 1:
            print('this value of n too small')
            exit(0)
        else:            
            X_train = X_train[:p.n_train]
            y_train = y_train[:p.n_train]
        
    
    #################################################################### FITTING ##############################################################
    
    if not p.model_type == 'rf':
        
        # fit model
        if p.model_type == 'linear_sta':
            s.w = X_train.T @ y_train / X_train.shape[0]
        elif 'mdl' in p.model_type:
            if p.model_type == 'mdl_orig':
                U, sv, Vh = npl.svd(X_train / np.sqrt(p.n_train))
                a = U.T @ y_train # / (np.sqrt(p.n_train) * p.noise_std)
                a = a[:sv.size]
                def mdl_loss(l):
                    return np.sum(np.square(a) / (1 + np.square(sv) / l) + np.log(1 + np.square(sv) / l))
                opt_solved = minimize(mdl_loss, x0=1e-10)
                s.lambda_opt = opt_solved.x
                s.loss_val = opt_solved.fun
                inv = npl.pinv(X_train.T @ X_train / p.n_train + s.lambda_opt * np.eye(p.num_features))
                s.w = inv @ X_train.T @ y_train / p.n_train
            elif p.model_type == 'mdl_m1':
                eigenvals, eigenvecs = npl.eig(X_train.T @ X_train)
                
                if p.dset == 'pmlb' and p.n_train > p.num_features + 1:
                    def estimate_sigma_unbiased():
                        m = LinearRegression(fit_intercept=False)
                        m.fit(X_train, y_train)
                        y_pred = m.predict(X_train)
                        return np.sum(np.square(y_train - y_pred)) / (p.n_train - p.num_features - 1)
                    p.noise_std = estimate_sigma_unbiased()
                
                var = p.noise_std**2
                def mdl1_loss(l):
                    inv = npl.pinv(X_train.T @ X_train + l * np.eye(p.num_features))
                    thetahat = inv @ X_train.T @ y_train
                    mse_norm = npl.norm(y_train - X_train @ thetahat)**2 / (2 * var)
                    theta_norm = npl.norm(thetahat)**2 / (2 * var)
                    eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
                    return mse_norm + theta_norm + eigensum
                opt_solved = minimize(mdl1_loss, x0=1e-10)
                s.lambda_opt = opt_solved.x
                s.loss_val = opt_solved.fun
                inv = npl.pinv(X_train.T @ X_train + s.lambda_opt * np.eye(p.num_features))
                s.w = inv @ X_train.T @ y_train
        else:
            if p.model_type == 'ols':
                m = LinearRegression(fit_intercept=False)
            elif p.model_type == 'lasso':
                m = Lasso(fit_intercept=False, alpha=p.reg_param)
            elif p.model_type == 'ridge':
                if p.reg_param == -1:
                    m = RidgeCV(fit_intercept=False, alphas=np.logspace(-3, 3, num=10, base=10))
                else:
                    m = Ridge(fit_intercept=False, alpha=p.reg_param)
            
            m.fit(X_train, y_train)
            if p.reg_param == -1 and p.model_type == 'ridge':
                s.lambda_opt = m.alpha_
            s.w = m.coef_
        
        
        # save df
        if p.model_type == 'ridge':
            S = X_train @ np.linalg.pinv(X_train.T @ X_train + p.reg_param * np.eye(X_train.shape[1])) @ X_train.T
            s.df1 = np.trace(S @ S.T)
            s.df2 = np.trace(2 * S - S.T @ S)
            s.df3 = np.trace(S)
        else:
            s.df1 = min(p.n_train, p.num_features)
            s.df2 = s.df1
            s.df3 = s.df1
        
        
        print('here!')
        # store predictions and things about w
        # s.H_trace = np.trace(H)
        s.wnorm = np.linalg.norm(s.w)
        s.num_nonzero = np.count_nonzero(s.w)
        s.preds_train = X_train @ s.w
        s.preds_test = X_test @ s.w
        
        
    
    elif p.model_type == 'rf':
        rf = RandomForestRegressor(n_estimators=p.num_trees, max_depth=p.max_depth)
        rf.fit(X_train, y_train)
        s.preds_train = rf.predict(X_train)
        s.preds_test = rf.predict(X_test)
        
    
    # set things
    s.train_mse = metrics.mean_squared_error(s.preds_train, y_train)
    s.test_mse = metrics.mean_squared_error(s.preds_test, y_test)    
        
    save(out_name, p, s)

    
if __name__ == '__main__':
    t0 = time.time()
    from params import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        if sys.argv[i+1] == 'True':
            setattr(p, sys.argv[i], t(True))            
        elif sys.argv[i+1] == 'False':
            setattr(p, sys.argv[i], t(False))
        else:
            setattr(p, sys.argv[i], t(sys.argv[i+1]))
    
    print('fname ', p._str(p))
    for key, val in p._dict(p).items():
        print('  ', key, val)
    print('starting...')
    fit(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')