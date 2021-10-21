# Defines the fundamental operations of our evaluator

import torch
import distributions as dist

### Language Types
Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Exp = (Atom, List)
Env = dict


# Define the default Env (i.e the core functions defined in our lisp-like
# FOPPL language)
#
# NOTE: As of now our two Data Structures are specifically indexed exclusively
#       by int() casted values.
#
# TODO: Implement the probabilistic keywords as well as the control-flow 
#       procedures.
#
def function_primitives() -> Env:
    """
    @returns: an Env containing mappings between the core functions of our 
    language and their implementations in our evaluator.
    """
    core = Env({
        # Basic Mathematical Operations
        '+': lambda x: primitive_op(x, torch.add),
        '-': lambda x: primitive_op(x, torch.sub),
        '*': lambda x: primitive_op(x, torch.mul),
        '/': lambda x: primitive_op(x, torch.div),
        'sqrt': lambda x: torch.sqrt(x[0]),  # Could be a mistake

        # Boolean
        '<': lambda x: x[0] < x[1],

        # Data Structures
        'vector': primitive_list,
        'hash-map': lambda x: {int(x[i]): x[i + 1] for i in range(len(x)) if i % 2 == 0},

        # Functions that operate on our Data Structures
        'get': lambda x: primitive_get(x),
        'put': primitive_put,
        'first': lambda x: x[0][0],
        'second': lambda x: x[0][1],
        'rest': lambda x: x[0][1:],
        'last': lambda x: x[0][x[0].size()[0] - 1],
        'append': primitive_append,

        # Distributions
        'normal': lambda args: dist.Normal(args),
        'beta': lambda args: dist.Beta(args),
        'exponential': lambda args: dist.Exponential(args),
        'uniform': lambda args: dist.UniformContinuous(args),
        'discrete': lambda args: dist.Categorical(args),

        # Matrix Math
        'mat-transpose': lambda args: args[0].T,
        'mat-tanh': lambda args: torch.tanh(args[0]),
        'mat-add': lambda args: mat_add(args),
        'mat-mul': lambda args: mat_mul(args),
        'mat-repmat': lambda args: args[0].repeat(int(args[2]), int(args[1])),

        # Probabilistic Forms
        # sample
        # observe
        'sample': lambda x: x[0].sample(),
        'observe': lambda x: x[1],  # TODO

    })
    return core

def mat_mul(args):
    return torch.mul(torch.unsqueeze(args[0], 1), torch.unsqueeze(args[1], 1).T)

def mat_add(args):
    return torch.add(args[0], args[1].T)


def primitive_list(args):
    # We can expect that all elements in args would be homogenous
    if isinstance(args[0], dist.Distribution):
        return args

    print(args)
    try:
        return torch.tensor(data=args)
    except:
        return args


def primitive_get(args):
    data_structure = args[0]
    key = args[1]

    if isinstance(data_structure, list):
        return data_structure[int(key)]

    if isinstance(data_structure, dict):
        return data_structure[int(key)]

    return data_structure[int(key)]


def primitive_op(array, func):
    # array is the array of inputs to be multiplied
    left = array[0]
    right = array[1]

    return func(left, right)


def primitive_put(args):
    """
    @arguments:
        datastructure: either a vector or hash-map which we want to index
        index: the index into the datastructure
        value: The value which we would like to insert into our datastructure
    @returns: the datastructure updated with the value at $index replaced with
              $value.
    """

    datastructure = args[0]
    index = args[1]
    value = args[2]

    datastructure[int(index)] = value
    return datastructure


def primitive_append(args):
    """
    @arguments:
        vector: the vector to which we would like to append $value
        value: the value which we would like to append
    @returns: the vector updated with value appended to the end
    """
    assert (len(args) == 2)
    vector = args[0]
    value = args[1]

    vec = torch.flatten(vector).tolist()
    vec.append(value)
    return torch.tensor(vec)
