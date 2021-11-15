import sys
import torch
import numpy as np
from queue import Queue
from daphne import daphne
from evaluator import evaluate
from threading import Thread


def send(message_queue, args):
    message_type = args["type"]
    logW = args["info"]

    if message_type == "start":
        p = Thread(
            target=threaded_evaluate,
            args=(message_queue, args["ast"], logW)
        )
        p.start()
    elif message_type == "continue":
        p = Thread(
            target=threaded_evaluate,
            args=(message_queue, None, logW),
            kwargs={'continuation_pair': (args["continuation"], args["continuation_arguments"])}
        )
        p.start()
    elif message_type == "sample":
        # print(f"putting sample message on queue")
        message_queue.put(args)
    elif message_type == "observe":
        # print(f"putting observe message on queue")
        message_queue.put(args)
    elif message_type == "return":
        # print(f"putting return message on queue")
        message_queue.put(args)
    else:
        print(f"Message type {message_type} not implemented")


# Evaluates an Expression
def threaded_evaluate(message_queue, ast, info, env=None, continuation_pair=None):
    if continuation_pair is not None:
        continuation, continuation_arguments = continuation_pair
        res = continuation(*continuation_arguments)
    else:
        res = evaluate(ast, env)('address_start', lambda x: x)

    if type(res) is not tuple:
        message = {
            "type": "return", "return_value": res, "info": info,
        }
        send(message_queue, message)
        return

    continuation, continuation_arguments, sigma = res
    while sigma["type"] is "proc":
        res = continuation(*continuation_arguments)
        if type(res) is not tuple:
            message = {
                "type": "return", "return_value": res, "info": info,
            }
            send(message_queue, message)
            return
        continuation, continuation_arguments, sigma = res

    message = {"type": sigma["type"], "alpha": sigma["alpha"]}
    if sigma["type"] == "sample":
        message.update({
            "continuation": continuation,
            "distribution": sigma["distribution"],
            "info": info,
        })
    elif sigma["type"] == "observe":
        message.update({
            "continuation": continuation,
            "distribution": sigma["distribution"],
            "observation": continuation_arguments[0],
            "info": info,
        })

    send(message_queue, message)
    return


def IS_Sampling(ast, particle_count):
    results = []
    log_weights = []
    message_queue = Queue()
    for particle in range(particle_count):
        args = {
            "type": "start",
            "ast": ast,
            "message_queue": message_queue,
            "info": torch.tensor(0.)
        }
        send(message_queue, args)

    processed_particles = 0
    while processed_particles < particle_count:
        message = message_queue.get()
        message_type = message["type"]
        if message_type == "sample":
            distribution = message["distribution"]
            args = {
                "type": "continue",
                "continuation": message["continuation"],
                "continuation_arguments": [distribution.sample()],
                "info": message["info"],
            }
            send(message_queue, args)
        elif message_type == "observe":
            distribution = message["distribution"]
            observation = message["observation"]
            args = {
                "type": "continue",
                "continuation": message["continuation"],
                "continuation_arguments": [observation],
                "info": message["info"] + distribution.log_prob(observation),
            }
            send(message_queue, args)
        elif message_type == "return":
            results.append(message["return_value"])
            log_weights.append(message["info"])
            processed_particles += 1
            # print(f"Particles remaining: {particle_count - processed_particles}")
        else:
            print("THIS MESSAGE CASE IS UNIMPLEMENTED!!")
    return log_weights, results


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    for i in range(1, 5):
        print(f"Running Daphne for Program {i}")
        exp = daphne(['desugar-hoppl-cps', '-i', '../hw6/programs/{}.daphne'.format(i)])
        # print(f"Sampling for Program {i}")
        for pc in [1, 10, 100, 1000, 10000, 100000]:
            print(f"Sampling for Program {i} with Particle Count {pc}")
            log_weights, samples = IS_Sampling(exp, pc)
            # Processing
            samples = torch.stack(samples)
            samples = samples.reshape((samples.shape[0], samples.size().numel() // samples.shape[0]))
            log_weights = torch.tensor(log_weights)
            log_Z = torch.logsumexp(log_weights, 0) - torch.log(torch.tensor(log_weights.shape[0], dtype=float))
            log_norm_weights = log_weights - log_Z
            weights = torch.exp(log_norm_weights).detach().numpy()
            weighted_samples = (torch.exp(log_norm_weights).reshape((-1, 1)) * samples.float()).detach().numpy()
            print('covariance: ', np.cov(samples.float().detach().numpy(), rowvar=False, aweights=weights))
            print('posterior mean:', weighted_samples.mean(axis=0))
