import itertools
from slurmpy import Slurm
import numpy as np
from copy import deepcopy
partition = 'low'

OUT_BASE = '/scratch/users/vision/yu_dl/raaz.rsk/mdl_sim_may/may11/'
PARAMS_BASE = {
    'out_dir': [OUT_BASE + 'test'],
    'seed': range(0, 2),    
    'num_features': [500],    
#     'n_train_over_num_features': [1e-2, 1e-1, 0.75, 0.9, 1, 1.5, 5, 7.5, 2e1, 4e1],
    'n_train_over_num_features': [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    
    'n_test': [5000],
    
    'dset': ['gaussian'], # gaussian, pmlb
    'dset_num': [0],
    
    'iid': ['iid'], # iid, clustered, spike, decay
    'cov_param': [2], #np.linspace(0, 4, 5),
    'beta_type': ['gaussian'], # one_hot, gaussian
    'noise_distr': ['gaussian'], # gaussian, t, gaussian_scale_var, thresh
    'noise_std': [1e-1], #0.001],
    'model_type': ['mdl_orig', 'mdl_m1', 'ridge'], #'mdl', linear_sta', 'ridge', 'ols', 'lasso'],      
    'reg_param': [0, 1e-2, 1e-1, 1, 1e1, -1], # make sure to always have reg_param 0!, 
}

PARAMS_IID = {
    'out_dir': [OUT_BASE + 'iid'],
    'iid': ['iid'],
}
PARAMS_DECAY = {
    'out_dir': [OUT_BASE + 'decay'],
    'iid': ['decay']
}
PARAMS_CLUSTERED = {
    'out_dir': [OUT_BASE + 'clustered'],
    'iid': ['clustered']
}
PARAMS_T = {
    'out_dir': [OUT_BASE + 't'],
    'noise_distr': ['t']
}
PARAMS_THRESH = {
    'out_dir': [OUT_BASE + 'thresh'],
    'noise_distr': ['thresh']
}
PARAMS_SCALE_VAR = {
    'out_dir': [OUT_BASE + 'gaussian_scale_var'],
    'noise_distr': ['gaussian_scale_var']
}

for param_settings in [PARAMS_IID, PARAMS_DECAY, PARAMS_CLUSTERED,
                       PARAMS_T, PARAMS_THRESH, PARAMS_SCALE_VAR]:
    params_to_vary = deepcopy(PARAMS_BASE)
    params_to_vary.update(param_settings)
    print(params_to_vary)
    

    # run
    s = Slurm("double descent", {"partition": partition, "time": "3-0"})
    ks = sorted(params_to_vary.keys())
    vals = [params_to_vary[k] for k in ks]
    param_combinations = list(itertools.product(*vals)) # list of tuples
    print(len(param_combinations))
    ks = np.array(ks)
    i_model_type = np.where(ks == 'model_type')[0][0]
    i_reg_param = np.where(ks == 'reg_param')[0][0]
    i_dset = np.where(ks == 'dset')[0][0]
    i_dset_num = np.where(ks == 'dset_num')[0][0]



    params_full = []
    for t in param_combinations:
        # remove reg_param for non-ridge and non-lasso
        if t[i_reg_param] != 0 and not t[i_model_type] in ['ridge', 'lasso']:
            pass

        # remove reg_param = 0 for ridge and lasso
        elif t[i_reg_param] == 0 and t[i_model_type] in ['ridge', 'lasso']:
            pass

        # remove dset_num for non-pmlb
        elif t[i_dset] is not 'pmlb' and t[i_dset_num] > 0:
            pass

        else:
            params_full.append(t)

    print('num calls', len(params_full))
    for p in params_full:
        print(p[i_reg_param], p[i_model_type])

    # iterate
    for i in range(len(params_full)):
        param_str = 'module load python; python3 ../linear_experiments/fit.py '
        for j, key in enumerate(ks):
            param_str += key + ' ' + str(params_full[i][j]) + ' '
        s.run(param_str)
    
    
# sweep different ways to initialize weights
'''
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/double_descent/linear_strange_gaussian'],
    'seed': range(0, 6),    
    'num_features': [1000],
    'n_train_over_num_features': [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    

    'dset': ['gaussian'],
#     'dset': ['pmlb'], # pblm, gaussian
#     'dset_num': range(0, 12), # only if using pmlb, 12 of these seem distinct
    
    'n_test': [5000],
    'noise_mult': [0.1], #0.001],
    'model_type': ['linear', 'linear_sta', 'linear_univariate'],     
#     'ridge_param': [0, 1e-2, 1e-1, 1],
}
'''

# sweep different things for linear sta
'''
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/double_descent/linear_sta'],
    'seed': range(0, 6),    
    'num_features': [1000],
    'n_train_over_num_features': [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    

#     'dset': ['gaussian'],
    'dset': ['pmlb'], # pblm, gaussian
    'dset_num': range(0, 2), # only if using pmlb, 12 of these seem distinct
    
    'n_test': [5000],
    'noise_mult': [0, 1e-2, 1e-1], #0.001],
    'model_type': ['linear_sta'],     
#     'ridge_param': [0, 1e-2, 1e-1, 1],
}
'''

'''
# run w/ diffeent covariances
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/double_descent/cov_vary'],
    'seed': range(0, 6),    
    'num_features': [1000],
    'n_train_over_num_features': [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    

#     'dset': ['gaussian'],
    'dset': ['gaussian'], # pblm, gaussian
    
    'n_test': [5000],
    'noise_mult': [1e-1], #0.001],
    'model_type': ['linear'],     
    'iid': [False],
#     'ridge_param': [0, 1e-2, 1e-1, 1],
}
'''

# for rf
'''
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/double_descent/rf_pmlb_sweep'],
    'seed': range(0, 7),    

    'num_features': [1000],
    'n_train_over_num_features': [5], # [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    

    
    'dset': ['pmlb'], # pmlb, gaussian
    'dset_num': range(0, 12), # only if using pmlb, 12 of these seem distinct
    
    'n_test': [5000],
    'noise_mult': [0.1], #0.001],
    'model_type': ['rf'],    
    'num_trees': [1, 2, 4, 8, 10, 16, 32, 64],
    'max_depth': [1, 2, 3, 4, 8, 10, 12]
}
'''
