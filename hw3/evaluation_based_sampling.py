import torch
from daphne import daphne
from process_results import process_results
from primitives import core
from tests import is_tol, load_truth, run_prob_test


def evaluate(exp, sigma, env):
    if type(exp) in [int, float]:
        return torch.tensor(float(exp)), sigma

    elif type(exp) == str:
        return env[exp], sigma

    op, *args = exp
    if op == 'sample':
        dist, sigma = evaluate(exp[1], sigma, env)
        return dist.sample(), sigma

    elif op == 'observe':
        fake_sample = exp[2]
        return evaluate(fake_sample, sigma, env)

    elif op == 'let':
        value, sigma = evaluate(args[0][1], sigma, env)
        env[args[0][0]] = value
        return evaluate(args[1], sigma, env)

    elif op == 'if':
        test, conseq, alt = args
        b, sigma = evaluate(test, sigma, env)
        return evaluate(conseq if b else alt, sigma, env)

    # Else call procedure:
    proc, sigma = evaluate(op, sigma, env)
    c = [0] * len(args)
    for (i, arg) in enumerate(args):
        c[i], sigma = evaluate(arg, sigma, env)
    return proc(c), sigma


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    sigma = {}
    env = core

    for (i, exp) in enumerate(ast):
        if type(exp) != list or exp[0] != 'defn':
            break
        env[exp[1]] = UserFunction(exp[2], exp[3], env)
    return evaluate(ast[i], sigma, env)[0]


class UserFunction:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, args):
        for (k, v) in zip(self.params, args):
            self.env[k] = v
        return evaluate(self.body, {}, self.env)[0]


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def run_deterministic_tests():
    for i in range(1, 15):
        # note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(ast)
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
        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)
        print(f'Probabilistic Test {i} passed')

    print('All probabilistic tests passed')


if __name__ == '__main__':
    run_deterministic_tests()
    run_probabilistic_tests()

    # Bins to use for the histograms of each of the programs
    bins = [50, 50, 3, 50]
    for i in range(1, 5):
        ast = daphne(['desugar', '-i', '../hw2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        stream = get_stream(ast)
        process_results(stream, f"{i}.daphne", "evaluation_based_sampler", bins[i - 1])
