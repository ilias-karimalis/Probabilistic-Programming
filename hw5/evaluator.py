from __future__ import division

import torch

from primitives import Normal, push_addr
from daphne import daphne
from tests import is_tol, run_prob_test, load_truth

## (c) Peter Norvig, 2010-16; See http://norvig.com/lispy.html


# Types
Symbol = str  # A Lisp Symbol is implemented as a Python str
List = list  # A Lisp List is implemented as a Python list
Atom = (int, float, bool)  # A Lisp Number is implemented as a Python int or float


# Environments
def standard_env():
    env = Env()
    # env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update({
        # Basic Mathematical Operations
        '+': lambda addr, arg0, arg1: arg0 + arg1,
        '-': lambda addr, arg0, arg1: arg0 - arg1,
        '*': lambda addr, arg0, arg1: arg0 * arg1,
        '/': lambda addr, arg0, arg1: arg0 / arg1,
        'sqrt': lambda addr, arg0: arg0 ** 0.5,
        'abs': lambda addr, arg0: abs(arg0),

        # Matrix Functions
        # 'mat-mul': lambda args: torch.matmul(args[0], args[1]),
        # 'mat-add': lambda args: torch.add(args[0], args[1]),
        # 'mat-transpose': lambda args: args[0].T,
        # 'mat-tanh': lambda args: torch.tanh(args[0]),
        # 'mat-repmat': lambda args: args[0].repeat(args[1].long(), args[2].long()),

        # Logic Operators
        '<': lambda addr, arg0, arg1: arg0 < arg1,
        '=': lambda addr, arg0, arg1: arg0 == arg1,
        'and': lambda addr, arg0, arg1: arg0 and arg1,
        'or': lambda addr, arg0, arg1: arg0 or arg1,

        # Data Structures
        'vector': lambda addr, *args: primitive_list(list(args)),
        'hash-map': lambda addr, *x: {int(x[i]): x[i + 1] for i in range(len(x)) if i % 2 == 0},

        # Functions that operate on our Data Structures
        'get': lambda addr, arg0, arg1: primitive_get(arg0, arg1),
        'put': lambda addr, arg0, arg1, arg2: primitive_put(arg0, arg1, arg2),
        'first': lambda addr, arg0: arg0[0],
        'second': lambda addr, arg0: arg0[1],
        'rest': lambda addr, arg0: arg0[1:],
        'last': lambda addr, arg0: arg0[-1],
        'append': lambda addr, arg0, arg1: primitive_append(arg0, arg1),

        # Distributions
        'normal': Normal,
        # 'discrete': lambda *args: Categorical(args),
        # 'gamma': lambda args: Gamma(args[0], args[1]),
        # 'dirichlet': lambda args: Dirichlet(args[0]),
        # 'flip': lambda args: Bernoulli(args[0]),
        # 'uniform-continuous': lambda args: Gamma((args[0] + args[1]) / 2, torch.tensor(1.0)),

        'push-address': push_addr,
    })
    return env


def primitive_list(args):
    try:
        return torch.stack(args)
    except:
        return args


def primitive_get(data_structure, key):
    if isinstance(data_structure, list):
        return data_structure[int(key)]
    elif isinstance(data_structure, dict):
        return data_structure[int(key)]
    return data_structure[int(key)]


def primitive_put(datastructure, index, value):
    datastructure[int(index)] = value
    return datastructure


def primitive_append(vector, value):
    if torch.is_tensor(vector):
        return torch.hstack([vector, value])
    return vector + value


class Env(dict):

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


global_env = standard_env()


# Procedures
class Procedure(object):

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return evaluate(self.body, Env(parms=self.parms, args=args, outer=self.env))


# eval
def evaluate(x, env=global_env):
    if isinstance(x, Atom):
        return torch.tensor(float(x))
    if isinstance(x, Symbol):
        try:
            return env.find(x)[x]
        except:
            return x


    op, *args = x
    if op == 'if':
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'fn':
        params, body = args[0], args[1]
        c = [""]
        if len(args) > 3:
            c = [evaluate(arg) for arg in args[2:]]

        return Procedure(params, body, env)(*c)
    else:
        proc = evaluate(x[0], env)
        args = [evaluate(exp, env) for exp in x[1:]]
        print(f"proc: {proc}")
        print(f"args: {args}")
        return proc(*args)


def run_deterministic_tests():
    # for i in range(1, 14):
    #
    #     exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/deterministic/test_{}.daphne'.format(i)])
    #     truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
    #     ret = evaluate(exp)
    #     try:
    #         assert (is_tol(ret, truth))
    #     except:
    #         raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))
    #     print(f"FOPPL Test {i} passed")
    #
    # print('FOPPL Tests passed')

    for i in range(10, 11):

        exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        # truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        print(exp)
        # ret = evaluate(exp)
        # try:
        #     assert (is_tol(ret, truth))
        # except:
        #     raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))
        #
        # print('Test passed')

    print('All deterministic tests passed')


# def run_probabilistic_tests():
#     num_samples = 1e4
#     max_p_value = 1e-2
#
#     for i in range(1, 7):
#         exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
#         truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
#
#         stream = get_stream(exp)
#
#         p_val = run_prob_test(stream, truth, num_samples)
#
#         print('p value', p_val)
#         assert (p_val > max_p_value)
#
#     print('All probabilistic tests passed')


if __name__ == '__main__':

    run_deterministic_tests()
    # run_probabilistic_tests()

    # for i in range(1, 13):
    #     # print(i)
    #     exp = daphne(['desugar-hoppl', '-i', '../hw5/programs/{}.daphne'.format(i)])
    #
    #
    #     print(f"Program {i} expression: {exp}\n\n\n")
    #
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate(exp))
