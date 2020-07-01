Official code for using / reproducing MDL-COMP from the paper "Rethinking complexity and the bias-variance tradeoff" ([arXiv link](https://arxiv.org/abs/2006.10189)). This code implements the calculation of MDL Complexity given training data and explores its ability to inform generalization. MDL-COMP is a complexity measure based on the principle of minimum description length of Rissanen. It enjoys nice theoretical properties and can be used to perform model selection, showing results on par with cross-validation (and sometimes even better with limited data).

*Note: this repo is actively maintained. For any questions please file an issue.*

# Reproducing the results in the paper
- most of the results can be produced by simply running the notebooks
- the experiments with real-data are more in depth and require running the `submit_real_data_jobs.py` file before running the notebook to view the analysis

![](https://csinva.github.io/mdl-complexity/results/fig_iid_mse.svg)


## Calculating MDL-COMP
Computation of `Prac-MDL-Comp` is fairly straightforward:

```python
import numpy.linalg as npl
import numpy as np
import scipy.optimize


def prac_mdl_comp(X_train, y_train, variance=1):
    '''Calculate prac-mdl-comp for this dataset
    '''
    eigenvals, eigenvecs = npl.eig(X_train.T @ X_train)

    def calc_thetahat(l):
        inv = npl.pinv(X_train.T @ X_train + l * np.eye(X_train.shape[1]))
        return inv @ X_train.T @ y_train

    def prac_mdl_comp_objective(l):
        thetahat = calc_thetahat(l)
        mse_norm = npl.norm(y_train - X_train @ thetahat)**2 / (2 * variance)
        theta_norm = npl.norm(thetahat)**2 / (2 * variance)
        eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
        return (mse_norm + theta_norm + eigensum) / y_train.size

    opt_solved = scipy.optimize.minimize(prac_mdl_comp_objective, x0=1e-10)
    prac_mdl = opt_solved.fun
    lambda_opt = opt_solved.x
    thetahat = calc_thetahat(lambda_opt)
    
    return {
        'prac_mdl': prac_mdl,
        'lambda_opt': lambda_opt,
        'thetahat': thetahat
    }
```

# Reference

- feel free to use/share this code openly
- if you find this code useful for your research, please cite the following:
```c
@article{dwivedi2020revisiting,
  title={Revisiting complexity and the bias-variance tradeoff},
  author={Dwivedi, Raaz and Singh, Chandan and and Yu, Bin and Wainwright, Martin},
  journal={arXiv preprint arXiv:2006.10189},
  year={2020}
}
```
