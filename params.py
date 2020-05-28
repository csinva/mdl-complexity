import numpy as np
from random import randint

class p:
    seed = 15
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/double_descent/test'
    dset = 'pmlb' # gaussian
    beta_type = 'gaussian' # one_hot
    beta_norm = 1
    iid = 'iid' # 'iid', 'clustered', 'spike', decay
    dset_num = 1 # only matters for pmlb
    dset_name = ''
    reg_param = -1.0 # -1 use csv
    num_features = 100
    n_train_over_num_features = 0.75 # 0.75 # this and num_features sets n_train
    n_test = 100
    noise_std = 1e-1
    noise_distr = 'gaussian' # gaussian, t, gaussian_scale_var, thresh
    model_type = 'mdl_m1' # mdl_orig, ridge
    cov_param = 0.0
    
    # for rf
    num_trees = 10
    max_depth = 10

    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + 'dset=' + str(vals['dset'])
    
class S:   
    def __init__(self, p):
        
        self.mean_max_corrs = {} # dict containing max_corrs, W_norms, mean_class_act
        # {mean_max_corrs: {it: {'fc.0.weight': val}}}
        
        # accs / losses
        self.train_mse = []
        self.test_mse = []
        self.wnorm = []
        self.pseudo_trace = []
        self.cov_trace = []
        self.nuclear_norm = []
        self.H_trace = []
        self.lambda_opt = None
        self.dset_name = None
    
    # dictionary of everything but weights
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()}