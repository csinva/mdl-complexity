import itertools
from slurmpy import Slurm
import numpy as np

partition = 'high'
kernel_version = False

params_to_vary = {
    'run': list(range(100)), # should be range(100)
}


# run
s = Slurm("fmri", {"partition": partition, "time": "2-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(len(param_combinations))
ks = np.array(ks)

# iterate
for i in range(len(param_combinations)):
    if kernel_version:
        param_str = '/usr/local/linux/anaconda3.7/bin/python ../fmri/run_kernel.py '
    else:
        param_str = 'module load python/3.8; which python; python ../fmri/run.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    print(param_str)
    s.run(param_str)
