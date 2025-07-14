import random
import numpy as np

def bootstrap(data_list, num_bootstrap=4, num_samples=None, seed=0):
    if num_samples is None:
        num_samples = len(data_list)
    means = []
    random.seed(seed)
    for i in range(num_bootstrap):
        exp_idxs = random.choices(list(range(num_samples)), k=num_samples)
        means.append(np.mean([data_list[di] for di in exp_idxs]))
    return np.std(means)