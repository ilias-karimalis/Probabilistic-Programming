from __future__ import division

import sys
import threading
import time

import numpy as np
import torch

from primitives import standard_env, Procedure
from daphne import daphne
from tests import is_tol, run_prob_test, load_truth

# Types
Symbol = str  # A Lisp Symbol is implemented as a Python str
List = list  # A Lisp List is implemented as a Python list
Atom = (int, float, bool)  # A Lisp Number is implemented as a Python int or float


class HOPPLSampler:

    def __init__(self, args):
        self.ast = [args["ast"], '""']
        self.max_time = args["max_time"] if "max_time" in args.keys() else 0

    def run(self):
        global_env = standard_env()
        samples = []
        start = time.time()
        while time.time() - start < self.max_time:
            samples.append(self.__evaluate(self.ast, global_env))
        return samples

    def run_itter(self):
        global_env = standard_env()
        return self.__evaluate(self.ast, global_env)

    def __evaluate(self, x, env):
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
            exp = (conseq if self.__evaluate(test, env) else alt)
            return self.__evaluate(exp, env)
        elif op == 'fn':
            params, body = args[0], args[1]
            return Procedure(params, body, env)
        elif op == 'sample':
            _addr, dist = [self.__evaluate(exp, env) for exp in args]
            return dist.sample()
        elif op == 'observe':
            _addr, _dist, obs = [self.__evaluate(exp, env) for exp in args]
            return obs
        else:
            proc = self.__evaluate(x[0], env)
            args = [self.__evaluate(exp, env) for exp in x[1:]]
            if isinstance(proc, Procedure):
                return proc(self.__evaluate, *args)
            return proc(*args)


def run_deterministic_tests():
    for i in range(1, 14):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        hoppl_sampler = HOPPLSampler({"ast": exp})
        ret = hoppl_sampler.run_itter()
        try:
            assert (is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))
        print(f"FOPPL Test {i} passed")
    print('FOPPL Tests passed')

    for i in range(1, 13):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        hoppl_sampler = HOPPLSampler({"ast": exp})
        ret = hoppl_sampler.run_itter()
        try:
            assert (is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))

        print(f'HOPPL Deterministic Test {i} passed')

    print('All deterministic tests passed')


def run_probabilistic_tests():
    max_p_value = 1e-2

    for i in range(1, 7):
        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        hoppl_sampler = HOPPLSampler({"ast": exp, "max_time": 10})
        samples = hoppl_sampler.run()

        p_val = run_prob_test(samples, truth)

        print('p value', p_val)
        assert (p_val > max_p_value)

    print('All probabilistic tests passed')

