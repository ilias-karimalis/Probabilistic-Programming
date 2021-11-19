import time

import dill
import numpy as np
from matplotlib import pyplot as plt
import torch
from queue import Queue
from torch.nn.functional import softmax

from evaluator import evaluate
from daphne import daphne


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done': True})  # wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    particle_count = len(particles)
    weights = np.exp(np.array([lw.detach() for lw in log_weights]))
    new_particle_indexes = np.random.choice(range(particle_count), particle_count, p=weights/np.sum(weights))
    new_particles = [particles[index] for index in new_particle_indexes]
    logZ = np.log(np.sum(weights)/particle_count)
    return logZ, new_particles


def SMC(n_particles, exp, program_name):
    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):
        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.

        particles.append(res)
        weights.append(logW)

    done = False
    smc_cnter = 0
    alpha_cur = "DEFAULT_VALUE"
    while not done:
        print(f'In SMC step {smc_cnter}, logZ: {sum(logZs)}')
        for i in range(n_particles):  
            res = run_until_observe_or_end(particles[i])

            if 'done' in res[2]:
                particles[i] = res[0]
                if i == 0:
                    done = True
                else:
                    assert done

            else:
                particles[i] = res
                sigma = res[2]
                weights[i] = sigma['d'].log_prob(sigma['c'])

                if i == 0:
                    alpha_cur = sigma['alpha']
                else:
                    assert(alpha_cur == sigma["alpha"])

        if not done:
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)

        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


def plot(samples, labels, n_particles, program_name, file):
    if samples.ndim == 1:
        plt.hist(samples, bins=50)
        plt.title(f"Histogram of {labels[0]} using SMC with {n_particles} particles")
        plt.savefig(f"plots/{program_name}_{labels[0]}_SMC_{n_particles}.png")
        plt.clf()
        print(f"Posterior Expectation: {np.average(particles)}", file=res_file)
        print(f"Posterior Variance: {np.var(particles)}", file=res_file)

    elif samples.ndim == 2:
        for j, label in enumerate(labels):
            print(f"Label: {label}", file=file)
            plt.hist(samples[:, j], bins=50)
            plt.title(f"Histogram of {label} using SMC with {n_particles} particles")
            plt.savefig(f"plots/{program_name}_{label}_SMC_{n_particles}.png")
            plt.clf()
            print(f"Posterior Expectation: {np.average(particles[:, j])}", file=file)
            print(f"Posterior Variance: {np.var(particles[:, j])}", file=file)


if __name__ == '__main__':
    labels = {
        1: ["geometric"],
        2: ["mu"],
        3: [f"data_point_{i}" for i in range(17)],
        4: ["mu"]
    }
    for i in range(1, 5):
        exp = daphne(['desugar-hoppl-cps', '-i', '../hw6/programs/{}.daphne'.format(i)])
        for pc in [1, 10, 100, 1000, 10000, 100000]:
            res_file = open(f"results/Program{i}_SMC_{pc}_output.txt", "w")
            print(f"Sampling for Program {i} with Particle Count {pc}")
            start = time.time()
            logZ, particles = SMC(pc, exp, f"Program{i}")
            print(f"Sampling took {time.time() - start} seconds")
            particles = np.array([p.detach().numpy() for p in particles])
            print(f"Evidence estimate: {logZ}", file=res_file)
            plot(particles, labels[i], pc, f"Program{i}", res_file)
