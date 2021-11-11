import time
from evaluator import evaluate
import torch
import numpy as np
import json
import threading
import os
from queue import Queue


def new_id():
    return time.time()


def get_IS_sample(ast):
    # init calc:
    # output = lambda x: x
    # res = evaluate(exp, env=None)('addr_start', output)
    # # TODO : hint, "get_sample_from_prior" as a basis for your solution

    particle_count = 100  # Temporary
    results = []
    log_weights = []
    message_queue = Queue()
    for particle in range(particle_count):
        args = {
            "type": "start",
            "ast": ast,
            "message_queue": message_queue,
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
                "continuation_arguments": [distribution.sample()],  # TODO: Seems wrong...
                "logW": message["logW"],
            }
            send(message_queue, args)
        elif message_type == "observe":
            distribution = message["distribution"]
            observation = message["observation"]
            args = {
                "type": "continue",
                "continuation": message["continuation"],
                "continuation_arguments": [observation],
                "logW": message["logW"] + distribution.log_prob(observation),
            }
            send(message_queue, args)
        elif message_type == "return":
            results.append(message["result"])
            log_weights.append(message["logW"])
            processed_particles += 1
        else:
            print("THIS MESSAGE CASE IS UNIMPLEMENTED!!")
    return log_weights, results


def send(message_queue, args):
    message_type = args["type"]

    if message_type == "start":
        evaluation_thread = threading.Thread(
            target=threaded_evaluate,
            args=(message_queue, args["ast"])
        )
        evaluation_thread.start()
    elif message_type == "continue":
        evaluation_thread = threading.Thread(
            target=threaded_evaluate,
            args=(message_queue, None),
            kwargs={'continuation_pair': (args["continuation"], args["continuation_arguments"])}
        )
        evaluation_thread.start()
    elif message_type == "sample":
        message_queue.put(args)
    elif message_type == "observe":
        message_queue.put(args)
    elif message_type == "return":
        message_queue.put(args)
    else:
        print(f"Message type {message_type} not implemented")


# Evaluates an Expression
def threaded_evaluate(message_queue, ast, env=None, continuation_pair=None):
    if continuation_pair is not None:
        continuation, continuation_arguments = continuation_pair
        res = continuation(*continuation_arguments)
    else:
        res = evaluate(ast, env)('addr_start', lambda x: x)

    if type(res) is not tuple:
        message = {
            "type": "return",
            "return_value": res,
            # TODO We need to somehow keep track of logW.
            # This may involve a structural change to the evaluator
            # Or perhaps a way for us to keep track of which particle number
            # we are currently dealing with.
            # Undecided as of now but, leaning heavily on the second cause
            # I'd rather not change the evaluator.
        }
        send(message_queue, message)
        return

    continuation, continuation_arguments, sigma = res
    # We now deal with the proc type which is kind of dumb
    # TODO: I don't like
    while sigma["type"] not in ["sample", "observe"]:
        assert (sigma["type"], "proc")
        continuation, continuation_arguments, sigma = continuation(*continuation_arguments)

    message = {"type": sigma["type"]}
    if sigma["type"] == "sample":
        message.update({
            "continuation": continuation,
            "distribution": sigma["distribution"],
        })
    elif sigma["type"] == "observe":
        message.update({
            "continuation": continuation,
            "distribution": sigma["distribution"],
            "observation": continuation_arguments[0],
        })

    send(message_queue, message)


if __name__ == '__main__':

    for i in range(1, 5):
        ast = daphne()
        with open('programs/{}.json'.format(i), 'r') as f:
            exp = json.load(f)
        print('\n\n\nSample of prior of program {}:'.format(i))
        log_weights = []
        values = []
        for i in range(10000):
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)

        log_weights = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0], values.size().numel() // values.shape[0]))
        log_Z = torch.logsumexp(log_weights, 0) - torch.log(torch.tensor(log_weights.shape[0], dtype=float))

        log_norm_weights = log_weights - log_Z
        weights = torch.exp(log_norm_weights).detach().numpy()
        weighted_samples = (torch.exp(log_norm_weights).reshape((-1, 1)) * values.float()).detach().numpy()

        print('covariance: ', np.cov(values.float().detach().numpy(), rowvar=False, aweights=weights))
        print('posterior mean:', weighted_samples.mean(axis=0))
