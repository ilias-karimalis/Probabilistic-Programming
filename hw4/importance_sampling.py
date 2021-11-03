import time

import torch
from primitives import core
import wandb


class UserFunction:
    def __init__(self, params, body, user_functions, global_env):
        self.params = params
        self.body = body
        self.env = global_env
        self.user_functions = user_functions

    def __call__(self, args, sigma, evaluator):
        for (k, v) in zip(self.params, args):
            self.env[k] = v
        return evaluator(self.body, sigma, {**self.env, **self.user_functions})


class LWSampler:

    def __init__(self, args):
        # Parse args
        self.ast = args["ast"]
        self.max_time = args["max_time"]

        # Misc
        self.global_env = core
        self.sigma = {'logW': 0}

        # Parse User Functions
        self.__parse_user_functions()

    def run(self):
        samples = []
        weights = []


        start = time.time()

        while time.time() - start < self.max_time:
            sample, sigma_new = self.__eval(self.ast, self.sigma.copy(), self.global_env)
            samples.append(sample)
            weights.append(sigma_new['logW'])

        wandb.finish()

        return samples, weights

    def __parse_user_functions(self):
        user_functions = {}

        i = 0
        for exp in self.ast:
            if type(exp) != list or exp[0] != 'defn':
                break
            user_functions[exp[1]] = UserFunction(exp[2], exp[3], user_functions, self.global_env)
            i += 1

        self.ast = self.ast[i]
        self.user_functions = user_functions

    def __eval(self, exp, sigma, env):
        if type(exp) in [int, float, bool]:
            return torch.tensor(float(exp)), sigma

        elif type(exp) == str:
            if exp in self.user_functions:
                return self.user_functions[exp], sigma
            else:
                return env[exp], sigma

        op, *args = exp
        if op == 'sample':
            dist, sigma = self.__eval(exp[1], sigma, env)
            return dist.sample(), sigma

        elif op == 'observe':
            dist, sigma = self.__eval(exp[1], sigma, env)
            observed_value, sigma = self.__eval(exp[2], sigma, env)
            if type(observed_value) is bool:
                observed_value = torch.tensor(1.0 if observed_value else 0.0)
            sigma['logW'] += dist.log_prob(observed_value)
            return observed_value, sigma

        elif op == 'let':
            value, sigma = self.__eval(args[0][1], sigma, env)
            env[args[0][0]] = value
            return self.__eval(args[1], sigma, env)

        elif op == 'if':
            test, conseq, alt = args
            b, sigma = self.__eval(test, sigma, env)
            return self.__eval(conseq if b else alt, sigma, env)

        proc, sigma = self.__eval(op, sigma, env)
        c = [0] * len(args)
        for (i, arg) in enumerate(args):
            c[i], sigma = self.__eval(arg, sigma, env)
        return proc(c, sigma, self.__eval) if proc in self.user_functions.values() else (proc(c), sigma)
