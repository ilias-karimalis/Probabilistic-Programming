from daphne import daphne
import matplotlib.pyplot as plt
import numpy as np
import time

import importance_sampling as isample

# Q3

# Importance Sampling:
start = time.time()

ast = daphne(['desugar', '-i', '../hw3/programs/3.daphne'])
print(f"Sampling for Program 3")
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

prob_equality = weighted_samples[1] / sum(weighted_samples)

print(f"probability of z[1] == z[2]: {prob_equality}")
posterior_ident = f"3.daphne_posterior_for_equality?_using_importance_sampling"
plt.bar(["True", "False"], [prob_equality, 1 - prob_equality])
plt.title(posterior_ident)
plt.savefig(f"plots/{posterior_ident}.png")
plt.clf()

elapsed = time.time() - start
print(f"Seconds Elapsed: {elapsed}")