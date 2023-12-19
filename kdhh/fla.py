import numpy as np
from scipy.stats import pearsonr


def fdc(func, xl, xu, n_var, n_samples, opt=None):

    samples = [np.random.uniform(xl,xu,n_var) for _ in range(n_samples)]

    # evaluate fitnesses
    fitnesses = np.array(list(map(func, samples)))

    # get indecies of best samples
    best_indecies = np.where(fitnesses==np.min(fitnesses))

    d = [] # distance list
    f = [] # fitness difference list
    for sample, fitness in zip(samples, fitnesses):

        di = np.linalg.norm(sample - opt)
        fi = fitness

        d.append(di)
        f.append(fi)

    return pearsonr(f, d)[0] # pearson correlation
