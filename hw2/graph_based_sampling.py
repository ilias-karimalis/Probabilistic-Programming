import numpy as np
import torch
from matplotlib import pyplot as plt

from daphne import daphne

from primitives import function_primitives #TODO
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:

### Language Types
Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Exp = (Atom, List)
Env = dict


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


env = function_primitives()

def deterministic_eval(exp):
    # exp is a variable reference
    if isinstance(exp, Symbol):
        return env[exp]
    # exp is a number literal
    elif isinstance(exp, Number):
        return torch.tensor(float(exp))

    ### exp is some kind of op
    op, *args = exp
    # exp is an if statement
    if op == 'if':
        (test, conseq, alt) = args
        new_exp = conseq if deterministic_eval(test) else alt
        return deterministic_eval(new_exp)
    # exp is a let statement
    elif op == 'let':
        (var, value), expr = args

        new_value = deterministic_eval(value)
        env[var] = new_value
        return deterministic_eval(expr)
    # exp is a call to function defined in env
    else:
        proc = deterministic_eval(exp[0])
        args = [deterministic_eval(arg) for arg in exp[1:]]
        return proc(args)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    #print(graph)
    graph_structure = graph[1]

    V = graph_structure['V']
    A = graph_structure['A']
    P = graph_structure['P']
    Y = graph_structure['Y']

    #print(f"Result {graph[2]}")
    result_nodes = graph[2]

    #print(f'V: {V}')
    #print(f'A: {A}')
    #print(f'P: {P}')
    #print(f'Y: {Y}')

    sorted_nodes = topologicalSort(V, A)
    for node in sorted_nodes:
        value = probabilistic_eval(P[node])
        env[node] = value

    return deterministic_eval(result_nodes)

def probabilistic_eval(exp):
    op, *args = exp
    if op == 'sample*':
        dist = deterministic_eval(args[0])
        return dist.sample()
    elif op == 'observe*':
        dist_arg, observed = args
        dist = deterministic_eval(dist_arg)
        return dist.sample()


# Topologically sorts the graph
def topologicalSort(vertices, edges):
    # Mark all the vertices as not visited
    visited = [False]*len(vertices)
    ret = []

    # Define helper functions
    def getVertexNumber(v, vertices):
        for (i, vert) in enumerate(vertices):
            if v == vert:
                return i

        raise Exception(f"Vertex {v} not in graph")

    def topoSortHelper(i, visited, ret, vertices, edges):
        # We are currently visiting vertex i
        visited[i] = True

        # We now visit all adjacent Vertices that have yet to be visited
        try:
            adjacent = edges[vertices[i]]
        except:
            adjacent = []

        for v in adjacent:
            if not visited[getVertexNumber(v, vertices)]:
                topoSortHelper(getVertexNumber(v, vertices), visited, ret, vertices, edges)

        ret.insert(0, vertices[i])

    # Perform Sort using helper function
    for i in range(len(vertices)):
        if not visited[i]:
            topoSortHelper(i, visited, ret, vertices, edges)

    return ret




def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples= 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../hw2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)

        names = graph[2]
        singleton = False
        if not isinstance(names, list):
            names = [names]
            singleton = True

        p_val = run_prob_test(stream, truth, num_samples, names, singleton)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    
    print("\nStarting Deterministic Tests:")
    #run_deterministic_tests()
    print("\nStarting Probabilistic Tests:")
    #run_probabilistic_tests()

    num_samples = 1e4

    for i in range(3,4):
        graph = daphne(['graph','-i','../hw2/programs/{}.daphne'.format(i)])
        print(graph)
        print('\n\n\nSample of prior of program {}:'.format(i))

        stream = get_stream(graph)

        names = graph[2]
        singleton = False

        if not isinstance(names, list):
            names = [names]
            singleton = True
        else:
            names = names[1:]

        samples = []
        for _ in range(int(num_samples)):
            samples.append(next(stream))

        samples = np.array([s.numpy() for s in samples])
        print(samples)

        for (j, name) in enumerate(names):
            #plt.subplot(len(names) , 1, i + 1)
            if singleton:
                plt.hist(samples, bins=3)
            else:
                plt.hist(samples[:, j], bins=10)
            plt.title(name)
            plt.savefig(f'./3_dapne_{name}.png')
            plt.clf()

        if singleton:
            print(f"{names[0]} mean is {samples.mean()}")
        else:
            for (n, name) in enumerate(names):
                print(f"{name} mean is {samples[:, n].mean()}")

        print(f"Finished plot prep for program {i}")

    