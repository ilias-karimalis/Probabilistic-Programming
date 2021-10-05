# Defines the fundamental operations of our evaluator

import torch

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
        '+': torch.add,
        '-': torch.sub,
        '*': torch.multiply,
        '/': torch.div,
        'sqrt': torch.sqrt,

        # Data Structures
        'vector': lambda *x: torch.tensor(x),
        'hash-map': lambda *x: {int(x[i]):x[i+1] for i in range(len(x)) if i%2==0},

        # Functions that operate on our Data Structures
        'get': lambda x, y: x[int(y)],
        'put': primitive_put,
        'first': lambda x: x[0],
        'last': lambda x: x[x.size()[0] - 1],
        'append': primitive_append

        # Control Flow
        # if
        # defn
        # let
        
        # Probabilistic Forms
        # sample
        # observe

    })
    return core

def primitive_put(datastructure, index, value):
    """
    @arguments:
        datastructure: either a vector or hash-map which we want to index
        index: the index into the datastructure
        value: The value which we would like to insert into our datastructure
    @returns: the datastructure updated with the value at $index replaced with
              $value.
    """
    datastructure[int(index)] = value
    return datastructure

def primitive_append(vector, value):
    """
    @arguments:
        vector: the vector to which we would like to append $value
        value: the value which we would like to append
    @returns: the vector updated with value appended to the end
    """
    vec = vector.squeeze().tolist()
    vec.append(value)
    return torch.tensor(vec)