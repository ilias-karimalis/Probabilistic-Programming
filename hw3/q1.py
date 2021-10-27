from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm

import importance_sampling as isample
from mh_gibbs_samping import mhgibbs
from process_results import weighted_avg_and_var

# Q1
print(f"Sampling for Program 1\n")

# Importance Sampling:
print("Importance Sampling:")
start = time.time()

ast = daphne(['desugar', '-i', '../hw3/programs/1.daphne'])
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

average, variance = weighted_avg_and_var(samples, np.exp(weights))
print(f"mu mean: {average}")
print(f"mu variance: {variance}")
posterior_ident = f"1.daphne_posterior_for_mu_using_importance_sampling"
plt.hist(samples, bins=80, weights=np.exp(weights))
plt.title(posterior_ident)
plt.savefig(f"plots/{posterior_ident}.png")
plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")


# MH within Gibbs:
print("MH within Gibbs Sampling:")
start = time.time()

graph = daphne(['graph', '-i', '../hw2/programs/1.daphne'])
num_samples = int(1e6)

samples = mhgibbs(graph, num_samples)
samples = np.array([s.numpy() for s in samples])

average = np.average(samples)
variance = np.var(samples)

print(f"mu mean: {average}")
print(f"mu variance: {variance}")
posterior_ident = f"1.daphne_posterior_for_mu_using_mhgibbs_sampling"
plt.hist(samples, bins=80)
plt.title(posterior_ident)
plt.savefig(f"plots/{posterior_ident}.png")
plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")

