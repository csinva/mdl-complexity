import itertools
import numpy as np
from copy import deepcopy
import config
import subprocess



#################################### Here we specify a bunch of different hyperparameters
OUT_BASE = config.test_dir
PARAMS_BASE = {
    'out_dir': [OUT_BASE + 'test'],
    'seed': range(3),    
    'num_features': [500],    
#     'n_train_over_num_features': [1e-2, 1e-1, 0.75, 0.9, 1, 1.5, 5, 7.5, 2e1, 4e1],
    'n_train_over_num_features': [5e-3, 1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    
    'n_test': [5000],
    
    'dset': ['gaussian'], # gaussian, pmlb
    'dset_num': [0],
    
    'iid': ['iid'], # iid, clustered, spike, decay
    'cov_param': [2], #np.linspace(0, 4, 5),
    'beta_type': ['gaussian'], # one_hot, gaussian
    'noise_distr': ['gaussian'], # gaussian, t, gaussian_scale_var, thresh
    'noise_std': [1e-1], # 1.0
    'model_type': ['ols', 'mdl_m1', 'ridge'], #'mdl_orig', 'mdl_m1',linear_sta', 'ridge', 'ols', 'lasso'],      
    'reg_param': [-1], # don't really need anything if you have -1, which is cv (must have -1!)
}


PARAMS_IID = {
    'out_dir': [OUT_BASE + 'iid'],
    'iid': ['iid'],
}
PARAMS_DECAY = {
    'out_dir': [OUT_BASE + 'decay'],
    'iid': ['decay'],
    'cov_param': [2]
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
PARAMS_SCALE_VAR = {
    'out_dir': [OUT_BASE + 'gaussian_scale_var'],
    'noise_distr': ['gaussian_scale_var']
}
PARAMS_PMLBS = [
    {
        'out_dir': [OUT_BASE + f'pmlb{i}'],
        'dset': ['pmlb'],
        'dset_num': [i],
        'noise_std': [1],
        'reg_param': [-1],
    } for i in range(0, 28) # 28
]
PARAMS_PMLBS_5FOLD = [
    {
        'out_dir': [OUT_BASE + f'pmlb{i}'],
        'dset': ['pmlb'],
        'dset_num': [i],
        'noise_std': [1],
        'reg_param': [-5],
    } for i in range(0, 28) # 28
]














########################################## Here we select which sets of hypeparameters we want to run
# PARAMS_PMLBS are the hyperparameters used to generate Table 1.
ALL_PARAMS = PARAMS_PMLBS # PARAMS_PMLBS_5FOLD #
#         [PARAMS_IID, PARAMS_DECAY, PARAMS_CLUSTERED,
#               PARAMS_T, PARAMS_THRESH, PARAMS_SCALE_VAR] #+ PARAMS_PMLBS
for param_settings in ALL_PARAMS:
    params_to_vary = deepcopy(PARAMS_BASE)
    params_to_vary.update(param_settings)
    print(params_to_vary)
    

    # run
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
        # remove reg_param !=-1 for non-ridge and non-lasso
        if t[i_reg_param] >= 0 and not t[i_model_type] in ['ridge', 'lasso']:
            pass

        # remove reg_param == 0 for ridge and lasso
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
        param_str = 'python3 ../src/fit.py '
        for j, key in enumerate(ks):
            param_str += key + ' ' + str(params_full[i][j]) + ' '
        subprocess.call(param_str, shell=True)
