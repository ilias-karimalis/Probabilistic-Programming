import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from hw3.mh_gibbs_samping import mhgibbs_max_time


def weighted_avg_and_var(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance

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


#def process_results(evaluator, program_stream, num_samples, program_name, evaluator_type, labels, n_bins=50, use_weights=False, bool_res=False):
def process_results(args):
    labels = args["labels"]
    evaluator = args["evaluator"]
    program_name = args["program_name"]
    
    # Generate samples
    samples, weights = generate_samples(args)

    # We can currently only handle a single boolean return value
    if args["bool_res"]:
        samples = generate_boolean_samples(samples, weights)
        prob = samples[1] / sum(samples)

        print(f"Posterior probability of {labels[0]}: {prob}")
        plt.bar(["True", "False"], [prob, 1 - prob])
        plt.title(f"Posterior probability of {labels[0]} using {evaluator}")
        plt.savefig(f"figures/{program_name}_{labels[0]}_{evaluator}.png")
        plt.clf()
        exit(1)


    # Handling numeric result values
    if samples.ndim == 1:
        average, variance = weighted_avg_and_var(samples, np.exp(weights))
        print(f"{labels[0]} mean: {average}")
        print(f"{labels[0]} variance: {variance}")
        posterior_ident = f"{program_name}_posterior_for_{labels[0]}_from_{evaluator_type}"
        plt.hist(samples, bins=n_bins, weights=np.exp(weights))
        plt.title(posterior_ident)
        plt.savefig(f"plots/{posterior_ident}.png")
        plt.clf()

    elif samples.ndim == 2:
        for j in range(samples.shape[1]):
            average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
            print(f"{labels[j]} mean: {average}")
            print(f"{labels[j]} variance: {variance}")
            posterior_ident = f"{program_name}_posterior_for_{labels[j]}_from_{evaluator_type}"
            plt.hist(samples[:, j], bins=n_bins, weights=np.exp(weights))
            plt.title(posterior_ident)
            plt.savefig(f"plots/{posterior_ident}.png")
            plt.clf()

    else:
        print(f"Plotting for results with {samples.ndim} dimensions is not yet supported.")


def generate_samples(args):
    evaluator = args["evaluator"]
    max_time = args["max_time"]

    samples = []
    weights = []

    start = time.time()

    if evaluator is "ImportanceSampling":
        program_stream = args["program_stream"]

        while True:
            if time.time() - start > max_time:
                break

            sample, weight = next(program_stream)
            samples.append(sample)
            weights.append(weight)

            samples = np.array([s.numpy() for s in samples])
            weights = np.array(weights)
    
    elif evaluator is "MHGibbs":
        graph = args["graph"]
        samples = mhgibbs_max_time(graph, max_time)
    
    else:
        print(f"ERROR: {evaluator} processing not implemented")
    
    samples = np.array([s.numpy() for s in samples])
    # Probably a mistake
    weights = np.array(weights) if args["use_weights"] else np.zeros_like(samples.size)
    return samples, weights
