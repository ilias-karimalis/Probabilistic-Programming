from daphne import daphne
from benchmarking import run_benchmark

labels = {
    1: ["mu"],
    2: ["slope", "bias"],
    3: ["z[1] == z[2]"],
    4: ["is-raining"],
    5: ["x", "y"]
}

max_time = 10
bins = [50, 50, 0, 0, 50]


for i in range(4, 6):

    # Importance Sampling
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
    # print(f"Starting Importance Sampling for Program {i}")
    # run_benchmark(is_args)

    # MH in Gibbs
    mh_args = {
        "labels": labels[i],
        "evaluator": "MHGibbs",
        "max_time": max_time,
        "n_bins": bins[i - 1],
        "bool_res?": True if i in [3, 4] else False,
        "program_name": f"Program{i}",
        "use_weights?": False,
        "graph": daphne(['graph', '-i', f'../hw3/programs/{i}.daphne'])
    }
    print(f"Starting MH in Gibbs for Program {i}")
    run_benchmark(mh_args)
