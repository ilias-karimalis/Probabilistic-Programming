# args dict with potential keys:
# labels, evaluator, program_name, n_bins, bool_res?, max_time, ast, graph, use_weights
import time
import numpy as np
from matplotlib import pyplot as plt

from utils import weighted_avg_and_var
from importance_sampling import LWSampler
from mh_gibbs_samping import MHGibbsSampler
from hmc_sampling import HMCSampler


def run_benchmark(args):
    labels = args["labels"]
    evaluator = args["evaluator"]
    program_name = args["program_name"]
    n_bins = args["n_bins"]

    # Generate samples
    samples, weights, joint_logs = generate_samples(args)

    if len(joint_logs) > 0:
        plt.plot(joint_logs)
        plt.title(f"Joint Log Trace using {evaluator}")
        plt.xlabel("Iteration")
        plt.savefig(f"plots/{program_name}_jlt_{evaluator}.png")
        plt.clf()

    if evaluator == "MHGibbs" or evaluator == "HMC":
        plt.title(f"Sample Trace for {program_name}")
        plt.xlabel("Iteration")
        if samples.ndim == 1:
            plt.plot(samples, label=labels[0])
        elif samples.ndim == 2:
            for (i, label) in enumerate(labels):
                plt.plot(samples[:, i], label=labels[i])
        plt.savefig(f"plots/{program_name}_sample_trace_{evaluator}.png")
        plt.clf()

    # We can currently only handle a single boolean return value
    if args["bool_res?"]:
        # samples = generate_boolean_samples(samples, weights)
        prob = np.average(np.array([float(s) for s in samples]), weights=np.exp(weights))
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

    elif samples.ndim == 2:
        covar = np.cov(samples.T, aweights=np.exp(weights), ddof=0)
        print(f"covariance: {covar}")

        for (j, label) in enumerate(labels):
            average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
            print(f"{label} mean: {average}")
            print(f"{label} variance: {variance}")
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

    if evaluator == "ImportanceSampling":
        lw_sampler = LWSampler(args)
        samples, weights = lw_sampler.run()

    elif evaluator == "MHGibbs":
        mhgibbs_sampler = MHGibbsSampler(args)
        samples, joint_logs = mhgibbs_sampler.run()
        if "burnin" in args.keys():
            samples = samples[args["burnin"]:]

    elif evaluator == "HMC":
        hmc_sampler = HMCSampler(args)
        samples, joint_logs = hmc_sampler.run()
        if "burnin" in args.keys():
            samples = samples[args["burnin"]:]
        joint_logs = [jl.item() for jl in joint_logs]

    else:
        print(f"ERROR: {evaluator} processing not implemented")

    samples = np.array([s.numpy() for s in samples])
    weights_shape = samples.shape if samples.ndim == 1 else samples[:, 0].shape
    weights = np.array(weights) if args["use_weights?"] else np.zeros(weights_shape)
    return samples, weights, joint_logs


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
