import torch
from primitives import core


def evaluate(exp, sigma, env, user_functions):
    if type(exp) in [int, float]:
        return torch.tensor(float(exp)), sigma

    if type(exp) == bool:
        return exp, sigma

    elif type(exp) == str:
        if exp in user_functions:
            return user_functions[exp], sigma
        else:
            return env[exp], sigma

    op, *args = exp
    if op == 'sample':
        dist, sigma = evaluate(exp[1], sigma, env, user_functions)
        return dist.sample(), sigma

    elif op == 'observe':
        dist, sigma = evaluate(exp[1], sigma, env, user_functions)
        observed_value, sigma = evaluate(exp[2], sigma, env, user_functions)
        if type(observed_value) is bool:
            observed_value = torch.tensor(1.0 if observed_value else 0.0)
        sigma['logW'] += dist.log_prob(observed_value)
        return observed_value, sigma

    elif op == 'let':
        value, sigma = evaluate(args[0][1], sigma, env, user_functions)
        env[args[0][0]] = value
        return evaluate(args[1], sigma, env, user_functions)

    elif op == 'if':
        test, conseq, alt = args
        b, sigma = evaluate(test, sigma, env, user_functions)
        return evaluate(conseq if b else alt, sigma, env, user_functions)

    # Else call procedure:
    # We need to differentiate UserFunctions from primitive functions
    proc, sigma = evaluate(op, sigma, env, user_functions)
    c = [0] * len(args)
    for (i, arg) in enumerate(args):
        c[i], sigma = evaluate(arg, sigma, env, user_functions)
    return proc(sigma, c) if proc in user_functions.values() else (proc(c), sigma)


def likelihood_weighting(ast):
    sigma = {'logW': 0}
    user_functions = {}
    global_env = core

    i = 0
    for exp in ast:
        if type(exp) != list or exp[0] != 'defn':
            break
        user_functions[exp[1]] = UserFunction(exp[2], exp[3], global_env, user_functions)
        i += 1

    res, sigma = evaluate(ast[i], sigma, global_env, user_functions)

    return res, sigma['logW']


class UserFunction:
    def __init__(self, params, body, global_env, user_functions):
        self.params = params
        self.body = body
        self.env = global_env
        self.user_functions = user_functions

    def __call__(self, sigma, args):
        for (k, v) in zip(self.params, args):
            self.env[k] = v
        return evaluate(self.body, sigma, self.env, self.user_functions)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield likelihood_weighting(ast)
