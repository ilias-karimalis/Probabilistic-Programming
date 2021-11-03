from daphne import daphne
from benchmarking import run_benchmark

labels = {
    1: ["mu"],
    2: ["slope", "bias"],
    3: ["z[1] == z[2]"],
    4: ["is-raining"],
    5: ["x", "y"]
}

max_time = 600
bins = [50, 50, 0, 0, 50]
burnin = [3000, 3000, 0, 0, 3000]
num_leaps = [30, 20, 0, 0, 20]
epsilon = [0.1, 0.1, 0, 0, 0.001]


for i in range(1, 6):

    # Importance Sampling
    print(f"Starting Importance Sampling for Program {i}")
    is_args = {
        "labels": labels[i],
        "evaluator": "ImportanceSampling",
        "max_time": max_time,
        "n_bins": bins[i-1],
        "bool_res?": True if i in [3, 4] else False,
        "program_name": f"Program{i}",
        "use_weights?": True,
        "ast": daphne(['desugar', '-i', f'../hw3/programs/{i}.daphne'])
    }
    run_benchmark(is_args)

    # # MH in Gibbs
    # print(f"Starting MH in Gibbs for Program {i}")
    # mh_args = {
    #     "labels": labels[i],
    #     "evaluator": "MHGibbs",
    #     "max_time": max_time,
    #     "n_bins": bins[i - 1],
    #     "bool_res?": True if i in [3, 4] else False,
    #     "program_name": f"Program{i}",
    #     "use_weights?": False,
    #     "graph": daphne(['graph', '-i', f'../hw3/programs/{i}.daphne'])
    # }
    # run_benchmark(mh_args)
    #
    # # Hamiltonian Monte Carlo
    # if i in [1, 2, 5]:
    #     print(f"Starting HMC for Program {i}")
    #     hmc_args = {
    #         "labels": labels[i],
    #         "evaluator": "HMC",
    #         "max_time": max_time,
    #         "n_bins": bins[i - 1],
    #         "bool_res?": False,
    #         "program_name": f"Program{i}",
    #         "use_weights?": False,
    #         "graph": daphne(['graph', '-i', f'../hw3/programs/{i}.daphne']),
    #         "num_leaps": num_leaps[i - 1],
    #         "epsilon": epsilon[i - 1],
    #         "burnin": burnin[i - 1]
    #     }
    #     run_benchmark(hmc_args)
