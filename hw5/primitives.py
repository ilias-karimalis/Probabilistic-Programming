import torch
import torch.distributions as dist
from pyrsistent import pmap, PMap


class Env(dict):

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


class Procedure(object):

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, eval_func, *args):
        return eval_func(self.body, Env(parms=self.parms, args=args, outer=self.env))


def standard_env():
    env = Env()
    env.update({
        # Basic Mathematical Operations
        '+': lambda addr, arg0, arg1: arg0 + arg1,
        '-': lambda addr, arg0, arg1: arg0 - arg1,
        '*': lambda addr, arg0, arg1: arg0 * arg1,
        '/': lambda addr, arg0, arg1: arg0 / arg1,
        'sqrt': lambda addr, arg0: arg0 ** 0.5,
        'abs': lambda addr, arg0: abs(arg0),

        # Logic Operators
        '<': lambda addr, arg0, arg1: arg0 < arg1,
        '>': lambda addr, arg0, arg1: arg0 > arg1,
        '=': lambda addr, arg0, arg1: arg0 == arg1,
        'and': lambda addr, arg0, arg1: arg0 and arg1,
        'or': lambda addr, arg0, arg1: arg0 or arg1,
        'log': lambda _addr, arg0: torch.log(arg0),

        # Data Structures
        'vector': lambda addr, *args: primitive_list(list(args)),
        'hash-map': p_hashmap,

        # Functions that operate on our Data Structures
        'get': lambda addr, arg0, arg1: primitive_get(arg0, arg1),
        'put': lambda addr, arg0, arg1, arg2: primitive_put(arg0, arg1, arg2),
        'first': lambda addr, arg0: arg0[0],
        'second': lambda addr, arg0: arg0[1],
        'rest': lambda addr, arg0: arg0[1:],
        'last': lambda addr, arg0: arg0[-1],
        'append': lambda addr, arg0, arg1: primitive_append(arg0, arg1),
        'empty?': primitive_empty,
        'cons': primitive_cons,
        'peek': lambda _addr, arg0: arg0[0],

        # Distributions
        'normal': Normal,
        'beta': Beta,
        'exponential': Exponential,
        'uniform-continuous': Uniform,
        'flip': Bernoulli,
        'discrete': Categorical,

        # Addressing
        'push-address': push_addr,
    })
    return env


def primitive_cons(_addr, ds, value):
    # if isinstance(ds, list):
    #     if isinstance()
    #     return [value] + ds
    if isinstance(ds, list) and len(ds) == 0:
        return value # Remember it's already a tensor
    return torch.hstack((value, ds))


def primitive_empty(_addr, ds):
    if isinstance(ds, list) or isinstance(ds, PMap):
        return len(ds) == 0
    # else we have a tensor
    return ds.size(0) == 0


def p_hashmap(_addr, *x):
    try:
        return pmap({int(x[i]): x[i + 1] for i in range(len(x)) if i % 2 == 0})
    except:
        return pmap({eval(x[i]): x[i + 1] for i in range(len(x)) if i % 2 == 0})


def primitive_list(args):
    try:
        return torch.stack(args)
    except:
        return args


def primitive_get(data_structure, key):
    if isinstance(data_structure, list):
        return data_structure[int(key)]
    elif isinstance(data_structure, PMap):
        try:
            return data_structure.get(int(key))
        except:
            return data_structure.get(eval(key))
    return data_structure[int(key)]


def primitive_put(datastructure, index, value):
    if isinstance(datastructure, PMap):
        try:
            datastructure = datastructure.set(int(index), value)
        except:
            datastructure = datastructure.set(eval(index), value)
    else:
        datastructure[int(index)] = value
    return datastructure


def primitive_append(vector, value):
    if torch.is_tensor(vector):
        return torch.hstack([vector, value])
    return vector + value


class Normal(dist.Normal):

    def __init__(self, _addr, loc, scale):

        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()

        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """

        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]

        return Normal(*ps)

    def log_prob(self, x):

        self.scale = torch.nn.functional.softplus(self.optim_scale)

        return super().log_prob(x)


class Beta(dist.Beta):

    def __init__(self, _addr, conc0, conc1):
        super().__init__(conc0, conc1)


class Exponential(dist.Exponential):

    def __init__(self, _addr, rate):
        super().__init__(rate)


class Uniform(dist.Uniform):

    def __init__(self, _addr, low, high):
        super().__init__(low, high)


class Bernoulli(dist.Bernoulli):

    def __init__(self, _addr, prob):
        super().__init__(probs=prob)


class Categorical(dist.Categorical):

    def __init__(self, _addr, probs):
        super().__init__(probs)


def push_addr(alpha, value):
    return alpha + value


