import time

import dill
import numpy as np
from matplotlib import pyplot as plt
import torch
from queue import Queue
from torch.nn.functional import softmax

from evaluator import evaluate
from daphne import daphne


# def SMC_Sampling(ast, particle_count):
#     particles = []
#     log_weights = []
#     log_evidences = []
#     message_queue = Queue()
#     for particle in range(particle_count):
#         args = {
#             "type": "start",
#             "ast": ast,
#             "message_queue": message_queue,
#             # Not sure what else is needed for info
#             "info": {
#                 "alpha": ''
#                 # TODO Might need more here
#             },
#         }
#         send(message_queue, args)
#
#     alpha_curr = "DEFAULT_VALUE"
#     processed_particles = 0
#     done = False
#     while not done:
#         message = message_queue.get()
#         message_type = message["type"]
#
#         if message_type == "sample":
#             distribution = message["distribution"]
#             args = {
#                 "type": "continue"
#             }
#
#     while processed_particles < particle_count:
#         message = message_queue.get()
#         message_type = message["type"]
#
#         if message_type == "sample":
#             distribution = message["distribution"]
#             args = {
#                 "type": "continue",
#                 "continuation": message["continuation"],
#                 "continuation_arguments": [distribution.sample()],
#                 "info": message["info"],
#             }
#             send(message_queue, args)
#
#         elif message_type == "observe":
#             particles.append(message["continuation"])
#             log_weights.append(message["distribution"].log_prob(message["observation"]))
#             processed_particles += 1
#             alpha = message["info"]["alpha"]
#
#             if processed_particles == 1:
#                 alpha_curr = alpha
#
#             if processed_particles < particle_count:
#                 assert(alpha_curr == alpha)
#
#             if processed_particles == particle_count:
#                 particle_count = 0
#                 log_evidence, new_particles = resample_particles(particles, log_weights)
#                 particles = []
#                 log_weights = []
#                 log_evidences.append(log_evidence)
#                 for particle in range(particle_count):
#                     continuation, continuation_args, sigma = new_particles[particle]
#                     args = {
#                         "type": "continue",
#                         "continuation": continuation,
#                         "continuation_arguments": continuation_args,
#                     }
#                     send(message_queue, args)
#
#
#             else:
#                 print("Oh no ...")
#
#         elif message_type == "return":
#             particles.append()
#             particle_count += 1
#     print(log_evidences)
#     log_evidence = sum(log_evidences)
#     return log_evidence, particles


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

    #unormalized_weights = torch.exp(torch.stack(log_weights))
    #normalized_weights = np.array([s.detach().numpy() for s in softmax(torch.stack(log_weights))])
    new_particle_indexes = np.random.choice(range(particle_count), particle_count, p=weights/np.sum(weights))
    new_particles = [particles[index] for index in new_particle_indexes]
    #logZ = torch.log(torch.sum(unormalized_weights)/particle_count)
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

    # can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    alpha_cur = "DEFAULT_VALUE"
    while not done:
        print('In SMC step {}, Zs '.format(smc_cnter), sum(logZs))
        for i in range(n_particles):  # Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])

            if 'done' in res[2]:  # this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  # and enforces everything to be the same as the first particle
                else:
                    assert done
                    # if not done:
                    #     raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                particles[i] = res
                sigma = res[2]
                weights[i] = sigma['d'].log_prob(sigma['c'])

                if i == 0:
                    alpha_cur = sigma['alpha']
                else:
                    assert(alpha_cur == sigma["alpha"])

        if not done:
            # resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)

        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


def plot(samples, labels, n_particles, program_name):
    if samples.ndim == 1:
        plt.hist(samples, bins=50)
        plt.title(f"Histogram of {labels[0]} using SMC with {n_particles} particles")
        plt.savefig(f"plots/{program_name}_{labels[0]}_SMC_{n_particles}.png")
        plt.clf()

    elif samples.ndim == 2:
        for j, label in enumerate(labels):
            plt.hist(samples[:, j], bins=50)
            plt.title(f"Histogram of {label} using SMC with {n_particles} particles")
            plt.savefig(f"plots/{program_name}_{label}_SMC_{n_particles}.png")
            plt.clf()


if __name__ == '__main__':
    labels = {
        1: ["geometric"],
        2: ["mu"],
        3: [f"data_point_{i}" for i in range(17)],
        4: ["mu"]
    }
    for i in range(3, 4):
        exp = daphne(['desugar-hoppl-cps', '-i', '../hw6/programs/{}.daphne'.format(i)])
        for pc in [100000]: #[10, 100, 1000, 10000, 100000]:
            res_file = open(f"results/Program{i}_SMC_{pc}_output.txt", "w")
            print(f"Sampling for Program {i} with Particle Count {pc}")
            start = time.time()
            logZ, particles = SMC(pc, exp, f"Program{i}")
            print(f"Sampling took {time.time() - start} seconds")
            particles = np.array([p.detach().numpy() for p in particles])
            print(f"Evidence estimate: {logZ}", file=res_file)
            print(f"Posterior Expectation: {np.average(particles)}", file=res_file)
            print(f"Posterior Variance: {np.var(particles)}", file=res_file)
            plot(particles, labels[i], pc, f"Program{i}")
