# args dict with potential keys:
# labels, evaluator, program_name, n_bins, bool_res?, max_time, ast, graph, use_weights
import time
import numpy as np
from matplotlib import pyplot as plt

from utils import weighted_avg_and_var
import importance_sampling as isampling
from mh_gibbs_samping import mhgibbs_max_time


def run_benchmark(args):
    labels = args["labels"]
    evaluator = args["evaluator"]
    program_name = args["program_name"]
    n_bins = args["n_bins"]

    # Generate samples
    samples, weights = generate_samples(args)

    # We can currently only handle a single boolean return value
    if args["bool_res?"]:
        samples = generate_boolean_samples(samples, weights)
        prob = samples[1] / sum(samples)

        print(f"Posterior probability of {labels[0]}: {prob}")
        plt.bar(["True", "False"], [prob, 1 - prob])
        plt.title(f"Posterior probability of {labels[0]} using {evaluator}")
        plt.savefig(f"plots/{program_name}_{labels[0]}_{evaluator}.png")
        plt.clf()
        return

    # Handling numeric result values
    if samples.ndim == 1:
        average, variance = weighted_avg_and_var(samples, np.exp(weights))
        print(f"{labels[0]} mean: {average}")
        print(f"{labels[0]} variance: {variance}")
        plt.hist(samples, bins=n_bins, weights=np.exp(weights))
        plt.title(f"Posterior probability of {labels[0]} using {evaluator}")
        plt.savefig(f"plots/{program_name}_{labels[0]}_{evaluator}.png")
        plt.clf()
        return

    if samples.ndim == 2:
        covar = np.cov(samples.T, aweights=np.exp(weights))
        print(f"covariance: {covar}")

        for (j, label) in enumerate(labels):
            average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
            print(f"{label} mean: {average}")
            print(f"{label} variance: {variance}")
            plt.hist(samples[:, j], bins=n_bins, weights=np.exp(weights))
            plt.title(f"Posterior probability of {label} using {evaluator}")
            plt.savefig(f"plots/{program_name}_{label}_{evaluator}.png")
            plt.clf()
        return

    print(f"Plotting for results with {samples.ndim} dimensions is not yet supported.")


def generate_samples(args):
    evaluator = args["evaluator"]
    max_time = args["max_time"]
    samples = []
    weights = []
    start = time.time()

    if evaluator == "ImportanceSampling":
        ast = args["ast"]
        program_stream = isampling.get_stream(ast)
        while True:
            if time.time() - start > max_time:
                break
            sample, weight = next(program_stream)
            samples.append(sample)
            weights.append(weight)
        samples = np.array([s.numpy() for s in samples])
        weights = np.array(weights)

    elif evaluator == "MHGibbs":
        graph = args["graph"]
        samples = mhgibbs_max_time(graph, max_time)
        samples = np.array([s.numpy() for s in samples])


    else:
        print(f"ERROR: {evaluator} processing not implemented")

    # Probably a mistake
    weights_shape = samples.shape if samples.ndim == 1 else samples[:, 0].shape
    weights = np.array(weights) if args["use_weights?"] else np.zeros(weights_shape)
    return samples, weights


def generate_boolean_samples(samples, weights):
    samples_res = [0, 0]
    weights_res = [0, 0]

    for (s, w) in zip(samples, weights):
        if s:
            samples_res[1] += 1
            weights_res[1] += np.exp(w)
        else:
            samples_res[0] += 1
            weights_res[0] += np.exp(w)

    weight_sum = np.sum(weights_res)
    weighted_samples = [0, 0]
    weighted_samples[0] = samples_res[0] * weights_res[0] / weight_sum
    weighted_samples[1] = samples_res[1] * weights_res[1] / weight_sum

    return weighted_samples