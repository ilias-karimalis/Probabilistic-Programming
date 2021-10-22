import torch
from daphne import daphne
from hw2.process_results import process_results
from primitives import core
from tests import is_tol, run_prob_test,load_truth

env = core


def deterministic_eval(exp):
    if type(exp) in [int, float]:
        return torch.tensor(float(exp))

    elif type(exp) == str:
        return env[exp]

    op, *args = exp
    if op == 'if':
        test, conseq, alt = args
        b = deterministic_eval(test)
        return deterministic_eval(conseq if b else alt)

    # Else call procedure:
    proc = deterministic_eval(op)
    c = [0] * len(args)
    for (i, arg) in enumerate(args):
        c[i] = deterministic_eval(arg)
    return proc(c)


def sample_from_joint(graph):
    # This function does ancestral sampling starting from the prior.
    graph_structure = graph[1]
    nodes = graph_structure['V']
    edges = graph_structure['A']
    link_functions = graph_structure['P']
    result_nodes = graph[2]

    sorted_nodes = topologicalSort(nodes, edges)
    for node in sorted_nodes:
        value = probabilistic_eval(link_functions[node])
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
        graph = daphne(['graph', '-i', '../hw2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')
        
        
if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()

    # Bins to use for the histograms of each of the programs
    bins = [50, 50, 3, 50]

    for i in range(1,5):
        graph = daphne(['graph','-i','../hw2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))

        stream = get_stream(graph)
        process_results(stream, f"{i}.daphne", "graph_based_sampler", bins[i-1])
    