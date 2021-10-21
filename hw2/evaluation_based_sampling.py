import numpy as np
import torch
from matplotlib import pyplot as plt

from daphne import daphne
from primitives import function_primitives
from tests import is_tol, load_truth, run_prob_test
from tqdm.auto import tqdm

### Language Types
Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Exp = (Atom, List)
Env = dict


# envs are a recursive structure as such we'll define them as a subclass of
# Dict, which records the parent from which it is based.
#
# Suppose we have: {Core} -> {Let1} -> {Let2}
# If we are searching for symbol x in Let2 and cannot find it, we will proceed
# by searching in Let1 (and then Core) recursively.

# Still not completely sure this is needed
class World(dict):
    def __init__(self, params=(), args=(), parent=None):
        dict.__init__(self, zip(params, args))
        self.parent = parent

    def __getitem__(self, key):
        try:
            value = dict.__getitem__(self, key)
        except:
            value = self.parent[key]
        return value


class UserFunction:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, args):
        for (k, v) in zip(self.params, args):
            self.env[k] = v
        return eval(self.body, self.env)


core = World(parent=function_primitives())


# TODO add sigma to evaluator
def eval(exp: Exp, env=core):
    """
    Evaluate an expression in an environment

    @arguments:
        exp: the currect Expression to be evaluated
        env: a Dictionary representing the defined symbol mappings for the exp
             being currently evaluated
    @returns: the evaluated Expression or a new Env
    """
    # exp is a variable reference
    if isinstance(exp, Symbol):
        return env[exp]
    # exp is a number literal
    elif isinstance(exp, Number):
        return torch.tensor(float(exp))

    # exp is some kind of op
    op, *args = exp
    # exp is an if statement
    if op == 'if':
        (test, conseq, alt) = args
        new_exp = conseq if eval(test, env) else alt
        return eval(new_exp, env)
    # exp is a let statement
    elif op == 'let':
        (var, value), expr = args

        new_value = eval(value, env)
        env[var] = new_value
        return eval(expr, env)
    # exp is a call to function defined in env
    else:
        proc = eval(exp[0], env)
        args = [eval(arg, env) for arg in exp[1:]]
        return proc(args)


def primitive_user_function(arg_list, body, env):
    
    return lambda x: eval(body, env.update(zip(arg_list, x[0])))


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    ast_rem, world = parse_user_functions(ast, core)
    res = eval(ast_rem, env=world)
    return res


def parse_user_functions(ast, world):
    ret = []
    for (index, node) in enumerate(ast):
        if node[0] != 'defn':
            ret = node
            break

        (_, func_name, arg_list, body) = node
        world[func_name] = UserFunction(arg_list, body, world)

    return ret, world


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def run_deterministic_tests():
    for i in tqdm(range(1, 15)):
        # note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert (is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast))

        print(f'Deterministic Test {i} passed')

    print('All deterministic tests passed')


def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-4

    for i in range(1, 7):
        # note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../hw2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        names = ast[-1]
        print(names)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)
        print(f'Probabilistic Test {i} passed')

    print('All probabilistic tests passed')


if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()

    num_samples = 1e4

    for i in range(2, 3):
        ast = daphne(['desugar', '-i', '../hw2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        stream = get_stream(ast)

        names = ["slope", "bias"]
        singleton = False

        if not isinstance(names, list):
            names = [names]
            singleton = True

        samples = []
        for _ in range(int(num_samples)):
            samples.append(next(stream))

        samples = np.array([s.numpy() for s in samples])
        print(samples)

        for (i, name) in enumerate(names):
            plt.subplot(len(names), 1, i + 1)
            if singleton:
                plt.hist(samples, bins=80)
            else:
                plt.hist(samples[:, i], bins=80)
            plt.title(name)
            print(f"Finished plot prep for program {i}")
        plt.tight_layout()
        plt.show()

        if singleton:
            print(f"{names[0]} mean is {samples.mean()}")
        else:
            for (n, name) in enumerate(names):
                print(f"{name} mean is {samples[:, i].mean()}")
