import torch
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import function_primitives

### Language Types
Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Exp = (Atom, List)
Env = dict

global_env = function_primitives()

def eval(exp: Exp, env=global_env) -> Exp:
    """
    Evaluate an expression in an environment

    @arguments:
        exp: the currect Expression to be evaluated
        env: a Dictionary representing the defined symbol mappings for the exp
             being currently evaluated
    @returns: the evaluated Expression
    """

    # exp is a variable reference
    if isinstance(exp, Symbol):
        return env[exp]
    # exp is a number literal
    elif isinstance(exp, Number):
        return torch.tensor(exp)
    # exp is a call to function defined in env
    else:
        proc = eval(exp[0], env)
        args = [eval(arg, env) for arg in exp[1:]]
        return proc(*args)
        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """        
    #print(ast)
    #print(eval(ast[0]))
    return eval(ast[0]), None



def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
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