import numpy as np
from scipy.stats import pearsonr


def fdc(func, xl, xu, n_var, n_samples):

    samples = [np.random.uniform(xl,xu,n_var) for _ in range(n_samples)]

    # evaluate fitnesses
    fitnesses = np.array(list(map(func, samples)))

    # get indecies of best samples
    best_indecies = np.where(fitnesses==np.min(fitnesses))


    d = [] # distance list
    f = [] # fitness difference list
    for sample, fitness in zip(samples, fitnesses):

        di = 0
        fi = 0

        # there might be multiple best solution
        # in this case, take average
        for best_index in best_indecies:

            best_fitness = fitnesses[int(best_index)] # best fitness value
            best_sample = samples[int(best_index)] # best sample

            di += np.linalg.norm(sample-best_sample) # add distance
            fi += fitness - best_fitness # add fitness difference

        di /= len(best_indecies) # get average
        fi /= len(best_indecies)
        d.append(di)
        f.append(fi)

    return pearsonr(f, d)[0] # pearson correlation
