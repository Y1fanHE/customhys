from customhys import hyperheuristic as hh
from customhys import benchmark_func as bf
from copy import copy
from fla import fdc
import numpy as np


parameters = dict(
    cardinality=10,  # Max. numb. of SOs in MHs, lvl:1
    cardinality_min=5,  # Min. numb. of SOs in MHs, lvl:1
    num_iterations=100,  # Iterations a MH performs, lvl:1
    num_agents=50,  # Agents in population,     lvl:1
    as_mh=True,  # HH sequence as a MH?,     lvl:2
    num_replicas=5,  # Replicas per each MH,     lvl:2
    num_steps=1,  # Trials per HH step,       lvl:2
    stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
    max_temperature=100,  # Initial temperature (SA), lvl:2
    min_temperature=1e-6,  # Min temperature (SA),     lvl:2
    cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
    temperature_scheme='fast',  # Temperature updating (SA),lvl:2
    acceptance_scheme='exponential',  # Acceptance mode,          lvl:2
    allow_weight_matrix=False,  # Weight matrix,            lvl:2
    trial_overflow=False,  # Trial overflow policy,    lvl:2
    learnt_dataset=None,  # If it is a learnt dataset related with the heuristic space
    repeat_operators=True,  # Allow repeating SOs inSeq,lvl:2
    verbose=False,  # Verbose process,          lvl:2
    learning_portion=0.37,  # Percent of seqs to learn  lvl:2
    solver='static')  # Indicate which solver use lvl:1

dime = 5
funs = [
    bf.Griewank(dime),
    bf.Rastrigin(dime),
    bf.Schwefel226(dime),
    bf.Salomon(dime),
    bf.Sphere(dime)
]

probs = [fun.get_formatted_problem() for fun in funs]
mh_base = [None for _ in range(10)]
rep = 5


initial_mh = None
for prob in probs:

    print("Solving problem", prob.get("func_name"), "_"*15)

    # fdc of problem
    prob_fdc = fdc(
        func = prob.get("function"),
        xl = prob.get("boundaries")[0],
        xu = prob.get("boundaries")[1],
        n_var = prob.get("dimension"),
        n_samples = 10000,
        opt = prob.get("optimum_solution")
    )

    # find mh of problem
    idx = int((prob_fdc+1) // 0.4)
    init_mh = mh_base[idx]

    print("\tfdc =", prob_fdc)
    print("\tinit_mh =", init_mh)

    # apply found mh as initial point
    hyp = hh.Hyperheuristic(
        heuristic_space="default.txt",
        problem=prob,
        parameters=parameters,
    )

    def run(rep_i):
        np.random.seed(1000+rep_i)
        mh, perf, _, _ = hyp.solve(save_steps=False,
                                   initial_solution=copy(init_mh))
        print("\t\tmh=", mh, ", fit=", perf, sep="")
        return mh, perf

    results = list(map(run, range(rep)))
    best_mh, best_perf = min(results, key=lambda res: res[-1])

    print("\tbest mh =", best_mh)
    print("\tbest fit =", best_perf)

    # store best mh
    mh_base[idx] = best_mh
