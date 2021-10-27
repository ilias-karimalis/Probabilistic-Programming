from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np
import time

import importance_sampling as isample
from mh_gibbs_samping import mhgibbs

# Q4

# Importance Sampling:

start = time.time()

ast = daphne(['desugar', '-i', '../hw3/programs/4.daphne'])
print(f"Sampling for Program 4")
stream = isample.get_stream(ast)

num_samples = int(1e5)
samples = [0, 0]
weights = [0, 0]

for i in range(int(num_samples)):
    sample, weight = next(stream)

    if sample.numpy():
        samples[1] += 1
        weights[1] += np.exp(weight.numpy())
    else:
        samples[0] += 1
        weights[0] += np.exp(weight.numpy())

weight_sum = np.sum(weights)
weighted_samples = [0, 0]
weighted_samples[0] = samples[0]*weights[0] / weight_sum
weighted_samples[1] = samples[1]*weights[1] / weight_sum

prob_raining = weighted_samples[1] / sum(weighted_samples)

print(f"probability of is-raining: {prob_raining}")
posterior_ident = f"4.daphne_posterior_for_is-raining_using_importance_sampling"
plt.bar(["True", "False"], [prob_raining, 1 - prob_raining])
plt.title(posterior_ident)
plt.savefig(f"plots/{posterior_ident}.png")
plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")


# MH within Gibbs:
print("MH within Gibbs Sampling:")
start = time.time()

graph = daphne(['graph', '-i', '../hw3/programs/4.daphne'])
num_samples = int(1e5)

samples = mhgibbs(graph, num_samples)
results = [0, 0]
for s in samples:
    if s.numpy():
        results[1] += 1
    else:
        results[0] += 1

prob_raining = results[1] / sum(results)

print(f"probability of is-raining: {prob_raining}")
posterior_ident = f"3.daphne_posterior_is-raining_using_mhgibbs_sampling"
plt.bar(["True", "False"], [prob_raining, 1 - prob_raining])
plt.title(posterior_ident)
plt.savefig(f"plots/{posterior_ident}.png")
plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")