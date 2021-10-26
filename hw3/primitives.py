import torch
from distributions import Normal, Beta, Exponential, UniformContinuous, Categorical, Distribution, Gamma, Dirichlet, \
    Bernoulli, Dirac

# These could all be made more strict by requiring that each operation receive
# a list of arguments of the correct length !!!
# This would probably require splitting them into separate functions for each
# core operation
core = {
    # Basic Mathematical Operations
    '+': lambda args: args[0] + args[1],
    '-': lambda args: args[0] - args[1],
    '*': lambda args: args[0] * args[1],
    '/': lambda args: args[0] / args[1],
    'sqrt': lambda args: args[0] ** 0.5,

    # Matrix Functions
    'mat-mul': lambda args: torch.matmul(args[0], args[1]),
    'mat-add': lambda args: torch.add(args[0], args[1]),
    'mat-transpose': lambda args: args[0].T,
    'mat-tanh': lambda args: torch.tanh(args[0]),
    'mat-repmat': lambda args: args[0].repeat(args[1].long(), args[2].long()),

    # Logic Operators
    '<': lambda args: args[0] < args[1],
    '=': lambda args: args[0] == args[1],
    'and': lambda args: args[0] and args[1],
    'or': lambda args: args[0] or args[1],

    # Data Structures
    'vector': lambda args: primitive_list(args),
    'hash-map': lambda x: {int(x[i]): x[i + 1] for i in range(len(x)) if i % 2 == 0},

    # Functions that operate on our Data Structures
    'get': lambda args: primitive_get(args),
    'put': lambda args: primitive_put(args),
    'first': lambda args: args[0][0],
    'second': lambda args: args[0][1],
    'rest': lambda args: args[0][1:],
    'last': lambda args: args[0][-1],
    'append': lambda args: primitive_append(args),

    # Distributions
    'normal': lambda args: Normal(args),
    'beta': lambda args: Beta(args),
    'exponential': lambda args: Exponential(args),
    'uniform': lambda args: UniformContinuous(args),
    'discrete': lambda args: Categorical(args),
    'gamma': lambda args: Gamma(args),
    'dirichlet': lambda args: Dirichlet(args),
    'flip': lambda args: Bernoulli(args),
    'dirac': lambda args: Dirac(args)

}


def primitive_list(args):
    if isinstance(args[0], Distribution):
        return args
    try:
        return torch.stack(args)
    except RuntimeError:
        return torch.hstack(args)


def primitive_get(args):
    assert(len(args) == 2)
    data_structure = args[0]
    key = args[1]
    if isinstance(data_structure, list):
        return data_structure[int(key)]
    elif isinstance(data_structure, dict):
        return data_structure[int(key)]
    return data_structure[int(key)]


def primitive_put(args):
    assert(len(args) == 3)
    datastructure = args[0]
    index = args[1]
    value = args[2]
    datastructure[int(index)] = value
    return datastructure


def primitive_append(args):
    assert (len(args) == 2)
    vector = args[0]
    value = args[1]
    if torch.is_tensor(vector):
        return torch.hstack([vector, value])
    return vector + value
