from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np
import time

import importance_sampling as isample
from process_results import weighted_avg_and_var

# Q5

# Importance Sampling:
start = time.time()

ast = daphne(['desugar', '-i', '../hw3/programs/5.daphne'])
print(f"Sampling for Program 5")
stream = isample.get_stream(ast)

num_samples = int(1e5)
samples = []
weights = []

for i in range(int(num_samples)):
    sample, weight = next(stream)
    samples.append(sample)
    weights.append(weight)

samples = np.array([s.numpy() for s in samples])
weights = np.array(weights)

labels = ["x", "y"]
for (j, label) in enumerate(labels):
    average, variance = weighted_avg_and_var(samples[:, j], np.exp(weights))
    print(f"{label} mean: {average}")
    print(f"{label} variance: {variance}")
    posterior_ident = f"5.daphne_posterior_for_{label}_using_importance_sampling"
    plt.hist(samples[:, j], bins=20, weights=np.exp(weights))
    plt.title(posterior_ident)
    plt.savefig(f"plots/{posterior_ident}.png")
    plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")
