from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm

import importance_sampling as isample
from mh_gibbs_samping import mhgibbs
from process_results import weighted_avg_and_var

# Q2
print(f"Sampling for Program 2\n")
labels = ["slope", "bias"]


# Importance Sampling:
print("Importance Sampling:")
start = time.time()

ast = daphne(['desugar', '-i', '../hw3/programs/2.daphne'])
stream = isample.get_stream(ast)

num_samples = int(1e6)
samples = []
weights = []

for i in tqdm(range(int(num_samples))):
    sample, weight = next(stream)
    samples.append(sample)
    weights.append(weight)

samples = np.array([s.numpy() for s in samples])
weights = np.array(weights)

covar = np.cov(samples.T, aweights=np.exp(weights))
print(f"covariance: {covar}")
for (j, label) in enumerate(labels):
    average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
    print(f"{label} mean: {average}")
    print(f"{label} variance: {variance}")
    posterior_ident = f"2.daphne_posterior_for_{label}_using_importance_sampling"
    plt.hist(samples[:, j], bins=20, weights=np.exp(weights))
    plt.title(posterior_ident)
    plt.savefig(f"plots/{posterior_ident}.png")
    plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")


# MH within Gibbs:
print("MH within Gibbs Sampling:")
start = time.time()

graph = daphne(['graph', '-i', '../hw3/programs/2.daphne'])
num_samples = int(1e5)

samples = mhgibbs(graph, num_samples)
samples = np.array([s.numpy() for s in samples])

covar = np.cov(samples.T)
print(f"covariance: {covar}")

for (j, label) in enumerate(labels):
    average = np.average(samples[:, j])
    variance = np.var(samples[:, j])
    print(f"mu mean: {average}")
    print(f"mu variance: {variance}")
    posterior_ident = f"2.daphne_posterior_for_{label}_using_mhgibbs_sampling"
    plt.hist(samples[:, j], bins=20)
    plt.title(posterior_ident)
    plt.savefig(f"plots/{posterior_ident}.png")
    plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")
