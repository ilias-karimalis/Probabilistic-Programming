import sys

import numpy as np
from matplotlib import pyplot as plt

import hoppl_eval
from daphne import daphne


def weighted_avg_and_var(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance


def run_benchmark(args):
    labels = args["labels"]
    evaluator = args["evaluator"]
    program_name = args["program_name"]
    n_bins = args["n_bins"]

    # Generate samples
    samples, weights, joint_logs = generate_samples(args)

    # Create Results file
    res_file = open(f"results/{program_name}_{evaluator}_output.txt", "w")

    if samples.ndim == 1:
        average, variance = weighted_avg_and_var(samples, np.exp(weights))
        print(f"{labels[0]} mean: {average}", file=res_file)
        print(f"{labels[0]} variance: {variance}", file=res_file)
        plt.hist(samples, bins=n_bins, weights=np.exp(weights))
        plt.title(f"Posterior probability of {labels[0]} using {evaluator}")
        plt.savefig(f"plots/{program_name}_{labels[0]}_{evaluator}.png")
        plt.clf()

    elif samples.ndim == 2:
        covar = np.cov(samples.T, aweights=np.exp(weights), ddof=0)
        print(f"covariance: {covar}", file=res_file)

        for (j, label) in enumerate(labels):
            average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
            print(f"{label} mean: {average}", file=res_file)
            print(f"{label} variance: {variance}", file=res_file)
            plt.hist(samples[:, j], bins=n_bins, weights=np.exp(weights))
            plt.title(f"Posterior probability of {label} using {evaluator}")
            plt.savefig(f"plots/{program_name}_{label}_{evaluator}.png")
            plt.clf()

    else:
        print(f"Plotting for results with {samples.ndim} dimensions is not yet supported.")


def generate_samples(args):
    evaluator = args["evaluator"]
    samples = []
    weights = []
    joint_logs = []

    if evaluator == "HOPPLSampler":
        hoppl_sampler = hoppl_eval.HOPPLSampler(args)
        samples = hoppl_sampler.run()

    else:
        print(f"ERROR: {evaluator} processing not implemented")

    samples = np.array([s.numpy() for s in samples])
    weights_shape = samples.shape if samples.ndim == 1 else samples[:, 0].shape
    weights = np.array(weights) if args["use_weights?"] else np.zeros(weights_shape)
    return samples, weights, joint_logs


def main():
    max_time = 1800
    labels = {
        1: ["geometric"],
        2: ["mu"],
        3: [f"data_point_{i}" for i in range(17)],
    }
    for i in range(1, 4):
        print(f"Running Daphne for Program {i}")
        args = {
            "labels": labels[i],
            "evaluator": "HOPPLSampler",
            "max_time": max_time,
            "n_bins": 50,
            "bool_res?": False,
            "program_name": f"Program{i}",
            "use_weights?": False,
            "ast": daphne(['desugar-hoppl', '-i', '../hw5/programs/{}.daphne'.format(i)])
        }
        print(f"Starting Inference for Program {i}")
        run_benchmark(args)


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    hoppl_eval.run_deterministic_tests()
    hoppl_eval.run_probabilistic_tests()
    main()
