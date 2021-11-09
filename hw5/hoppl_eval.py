from __future__ import division

import sys
import threading
import time

import numpy as np
import torch

from primitives import Env, standard_env
from daphne import daphne
from tests import is_tol, run_prob_test, load_truth

# Types
Symbol = str  # A Lisp Symbol is implemented as a Python str
List = list  # A Lisp List is implemented as a Python list
Atom = (int, float, bool)  # A Lisp Number is implemented as a Python int or float


class HOPPLSampler:

    def __init__(self, args):
        self.ast = args["ast"]
        self.max_time = args["max_time"]


# Procedures
class Procedure(object):

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return evaluate(self.body, Env(parms=self.parms, args=args, outer=self.env))


# eval
def evaluate(x, env):
    if isinstance(x, Atom):
        return torch.tensor(float(x))
    if isinstance(x, Symbol):
        try:
            return env.find(x)[x]
        except:  # This is Hacky!!!
            return x

    op, *args = x
    if op == 'if':
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'fn':
        params, body = args[0], args[1]
        return Procedure(params, body, env)
    elif op == 'sample':
        _addr, dist = [evaluate(exp, env) for exp in args]
        return dist.sample()
    elif op == 'observe':
        _addr, _dist, obs = [evaluate(exp, env) for exp in args]
        return obs
    else:
        proc = evaluate(x[0], env)
        args = [evaluate(exp, env) for exp in x[1:]]
        return proc(*args)


def get_stream(ast):
    global_env = standard_env()
    while True:
        yield evaluate([ast, '""'], global_env)


def run_deterministic_tests():
    global_env = standard_env()

    for i in range(1, 14):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate([exp, ""], global_env)
        try:
            assert (is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))
        print(f"FOPPL Test {i} passed")
    print('FOPPL Tests passed')

    for i in range(1, 13):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate([exp, '""'], global_env)
        try:
            assert (is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))

        print(f'HOPPL Deterministic Test {i} passed')

    print('All deterministic tests passed')


def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-2

    for i in range(1, 7):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print(exp)
        stream = get_stream(exp)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)

    print('All probabilistic tests passed')


def main():
    # run_deterministic_tests()
    # run_probabilistic_tests()

    for i in range(1, 4):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/{}.daphne'.format(i)])
        # print(f"Program {i} expression: {exp}\n\n\n")
        print('Sample of prior of program {}:'.format(i))
        stream = get_stream(exp)
        max_time = 60
        samples = []
        start = time.time()
        while time.time() - start < max_time:
            samples.append(next(stream))
        samples = np.array([s.numpy() for s in samples])
        print(np.average(samples))


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    # threading.stack_size(200000)
    thread = threading.Thread(target=main)
    thread.start()
