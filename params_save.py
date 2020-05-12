import numpy as np
from numpy.random import randint

class S:   
    def __init__(self, p):
        
        self.mean_max_corrs = {} # dict containing max_corrs, W_norms, mean_class_act
        # {mean_max_corrs: {it: {'fc.0.weight': val}}}
        
        # accs / losses
        train_mse, test_mse, wnorm, pseudo_trace, cov_trace, nuclear_norm, H_trace = [], [], [], [], [], [], []
        
        self.lambda_opt = None
        
    
    # dictionary of everything but weights
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()}