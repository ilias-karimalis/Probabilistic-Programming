from re import X
import torch

from daphne import daphne
from primitives import function_primitives, primitive_append, primitive_put
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
class World(dict):
    def __init__(self, params=(), args=(), parent=None):
        dict.__init__(self, zip(params, args))
        self.parent = parent

    def __getitem__(self, key):
        try:
            value = dict.__getitem__(key)
        except:
            value = self.parent[key]
        return value        

class UserFunction:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env
    
    def __call__(self, *args):
        return eval(self.body, Env(self.params, args, self.env))

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


# TODO: Rework eval to return exp, env pairs
def eval(exp: Exp, env=core):
    """
    Evaluate an expression in an environment

    @arguments:
        exp: the currect Expression to be evaluated
        env: a Dictionary representing the defined symbol mappings for the exp
             being currently evaluated
    @returns: the evaluated Expression or a new Env
    """

    # We must now decide when to form a new World and when to propagate up info

    # exp is a variable reference
    if isinstance(exp, Symbol):
        return env[exp]
    # exp is a number literal
    elif isinstance(exp, Number):
        return torch.tensor(exp)

    ### exp is some kind of op
    op, *args = exp
    # exp is an if statement
    if op == 'if':
        (test, conseq, alt) = args
        new_exp = conseq if eval(test, env) else alt
        return eval(new_exp, env)
    # exp is a let statement
    elif op == 'let':
        (var, value), expr = args
        env[var] = eval(value, env)
        return eval(expr, env)
    # exp is a call to function defined in env
    else:
        print(exp)
        proc = eval(exp[0], env)
        args = [eval(arg, env) for arg in exp[1:]]
        return proc(*args)

def primitive_user_function(arg_list, body, env):
    print(arg_list)
    print(body)
    return lambda *x: eval(body, env.update(zip(arg_list, x)))

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """        
    print(ast)

    print(core['+'])

    ast_rem, world = parse_user_functions(ast, core)
    res = eval(ast_rem, env=world)
    return res, None

def parse_user_functions(ast, world):
    ret = []
    for (index, node) in enumerate(ast):
        
        if node[0] != 'defn':
            ret = node
            break

        (_, func_name, arg_list, body) = node
        world[func_name] = UserFunction(arg_list, body, world)

        ast.pop(index)
    return ret, world



def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in tqdm(range(1,15)):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../hw2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    #run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../hw2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])
