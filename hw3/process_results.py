import numpy as np
import matplotlib.pyplot as plt
import torch


def weighted_avg_and_var(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance


def process_results(program_stream, program_name, evaluator_type, labels, n_bins=50, use_weights=False):
    num_samples = int(1e5)
    samples = []
    weights = []

    for i in range(int(num_samples)):
        sample, weight = next(program_stream)
        samples.append(sample)
        weights.append(weight)

    samples = np.array([s.numpy() for s in samples])
    weights = np.array(weights)

    if type(samples[0]) is np.bool_:
        s_new = []
        for i in range(int(num_samples)):
            s_new.append(1. if samples[i] else 0.)
        samples = np.array(s_new)

    if not use_weights:
        weights = np.zeros_like(weights)

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

    elif samples.ndim == 3:
        for j in range(samples.shape[1]):
            for k in range(samples.shape[2]):
                print(f"{program_name}_means_for_result_({j},{k})_from_{evaluator_type}: {np.mean(samples[:, j, k])}")
                plt.hist(samples[:, j, k], bins=n_bins)
                plt.title(f"{program_name}_output_for_result_({j},{k})_from_{evaluator_type}")
                plt.savefig(f"plots/{program_name}_output_for_result_({j},{k})_from_{evaluator_type}.png")
                plt.clf()

    else:
        print(f"Plotting for results with {samples.ndim} dimensions is not yet supported.")


def produce_weighted_samples(samples, weights, use_weights):
    if use_weights:
        exp_weights = np.exp(weights)
        return np.multiply(samples, exp_weights) / np.sum(exp_weights)

    return samples
