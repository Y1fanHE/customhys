from customhys import hyperheuristic as hh
from customhys import benchmark_func as bf
from fla import fdc
import numpy as np


parameters = dict(
    cardinality=10,  # Max. numb. of SOs in MHs, lvl:1
    cardinality_min=5,  # Min. numb. of SOs in MHs, lvl:1
    num_iterations=100,  # Iterations a MH performs, lvl:1
    num_agents=30,  # Agents in population,     lvl:1
    as_mh=True,  # HH sequence as a MH?,     lvl:2
    num_replicas=30,  # Replicas per each MH,     lvl:2
    num_steps=50,  # Trials per HH step,       lvl:2
    stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
    max_temperature=100,  # Initial temperature (SA), lvl:2
    min_temperature=1e-6,  # Min temperature (SA),     lvl:2
    cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
    temperature_scheme='fast',  # Temperature updating (SA),lvl:2
    acceptance_scheme='exponential',  # Acceptance mode,          lvl:2
    allow_weight_matrix=True,  # Weight matrix,            lvl:2
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

mh_base = list()


initial_mh = None
for prob in probs:

    print("Solving problem", prob.get("func_name"), "_"*15)

    # fdc of problem
    prob_fdc = fdc(
        prob.get("function"),
        prob.get("boundaries")[0],
        prob.get("boundaries")[1],
        prob.get("dimension"),
        5000
    )

    # find mh of problem with closest fdc
    diff = 100
    init_mh = None
    for mh_fdc, mh in mh_base:
        if abs(mh_fdc - prob_fdc) < diff:
            diff = abs(mh_fdc - prob_fdc)
            init_mh = mh

    print("\tfdc of the problem is", prob_fdc)
    print("\tselect mh", init_mh, "as initial point")

    # apply found mh as initial point
    hyp = hh.Hyperheuristic(
        heuristic_space="default.txt",
        problem=prob,
        parameters=parameters,
        file_label=f'{prob.get("func_name")}-5D-Exp1'
    )
    best_mh, best_perf, hist_curr, hist_best = hyp.solve(initial_solution=init_mh)

    print("\tbest mh is", best_mh, "best fit is", best_perf)

    # store best mh and fdc
    mh_base.append([prob_fdc, best_mh])

    print("\tcurrent mh base")
    for prob_fdc, mh in mh_base:
        print("\t\t", prob_fdc, "->".join([str(i) for i in mh]))
