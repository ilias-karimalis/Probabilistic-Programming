import torch
from daphne import daphne
from primitives import core
from tests import is_tol, run_prob_test,load_truth
from tqdm.auto import tqdm
from distributions import Normal


def deterministic_eval(exp, env):
    if type(exp) in [int, float]:
        return torch.tensor(float(exp))

    elif type(exp) == str:
        return env[exp]

    op, *args = exp
    if op == 'if':
        test, conseq, alt = args
        res_conseq = deterministic_eval(conseq, env)
        res_alt = deterministic_eval(alt, env)
        b = deterministic_eval(test, env)
        return res_conseq if b else res_alt

    # Else call procedure:
    proc = deterministic_eval(op, env)
    c = [0] * len(args)
    for (i, arg) in enumerate(args):
        c[i] = deterministic_eval(arg, env)
    return proc(c)


def sample_from_joint(graph):
    # This function does ancestral sampling starting from the prior.
    local_env = core
    graph_structure = graph[1]
    nodes = graph_structure['V']
    edges = graph_structure['A']
    link_functions = graph_structure['P']
    result_nodes = graph[2]

    sorted_nodes = topologicalSort(nodes, edges)
    for node in sorted_nodes:
        value = probabilistic_eval(link_functions[node])
        local_env[node] = value

    return deterministic_eval(result_nodes, local_env)


def sample_from_priors(graph):
    local_env = {}
    graph_structure = graph[1]
    nodes = graph_structure['V']
    edges = graph_structure['A']
    link_functions = graph_structure['P']

    sorted_nodes = topologicalSort(nodes, edges)
    for node in sorted_nodes:
        value = probabilistic_eval(link_functions[node], {**core, **local_env})
        local_env[node] = value

    return local_env


def probabilistic_eval(exp, env):
    op, *args = exp
    if op == 'sample*':
        dist = deterministic_eval(args[0], env)
        return dist.sample()
    elif op == 'observe*':
        _, observed = args
        return deterministic_eval(observed, env)


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


def generate_markov_blankets(nodes, observed, edges):

    # Terrible list flattening solution I found on StackOverflow
    def list_flatten(arr):
        if type(arr) == list:
            return sum(map(list_flatten, arr), [])
        return [arr]

    # Generate Markov Blankets:
    markov_blankets = {}

    for node in nodes:
        if node not in observed:
            node_blanket = [node]

            # Add children
            node_blanket.extend(edges[node])

            markov_blankets[node] = node_blanket

    return markov_blankets


def mhgibbs(graph, num_samples):

    nodes = graph[1]['V']
    link_functions = graph[1]['P']
    edges = graph[1]['A']
    observed = graph[1]['Y']

    sampled = get_sampled(nodes, link_functions)
    markov_blankets = generate_markov_blankets(nodes, link_functions, observed, edges)

    samples = [sample_from_priors(graph)]
    for s in tqdm(range(num_samples)):
        sample = mhgibbs_step(link_functions, sampled, observed, samples[s], markov_blankets)
        samples.append(sample)

    # Samples here is a list where each sample is a dictionary mapping of r.v. to value
    # We should only return the ones needed for the result
    result_nodes = graph[2]
    samples = [deterministic_eval(result_nodes, {**core, **sample, **observed}) for sample in samples]
    return samples


def mhgibbs_step(link_functions, sampled, observed, last_sample, markov_blankets):
    sample = last_sample.copy()
    for rv in sampled:
        # Draw Samples from Prior for proposal
        dist = deterministic_eval(link_functions[rv][1], {**core, **last_sample})
        proposed_sample = sample.copy()
        proposed_sample[rv] = dist.sample()
        u = torch.rand(1)
        alpha = mhgibbs_acceptance(link_functions, rv, observed, proposed_sample, sample, markov_blankets[rv])
        if u < alpha:
            sample = proposed_sample

    return sample


def get_sampled(nodes, link_functions):
    sampled = []
    for node in nodes:
        if link_functions[node][0] == "sample*":
            sampled.append(node)

    return sampled


def mhgibbs_acceptance(link_functions, rv, observed, proposed_sample, last_sample, markov_blanket):
    dist_new = deterministic_eval(link_functions[rv][1], {**core, **proposed_sample})
    dist_last = deterministic_eval(link_functions[rv][1], {**core, **last_sample})
    log_alpha = dist_new.log_prob(last_sample[rv]) - dist_last.log_prob(proposed_sample[rv])
    for node in markov_blanket:
        log_alpha += deterministic_eval(
            link_functions[node][1],
            {**core, **observed, **proposed_sample}
        ).log_prob(proposed_sample[node])
        log_alpha -= deterministic_eval(
            link_functions[node][1],
            {**core, **observed, **last_sample}
        ).log_prob(last_sample[node])

    return torch.exp(log_alpha)


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
        
#
# if __name__ == '__main__':
#
#     # Bins to use for the histograms of each of the programs
#     bins = [50, 50, 3, 50]
#
#     for i in range(1,2):
#         graph = daphne(['graph','-i','../hw2/programs/{}.daphne'.format(i)])
#         print('\n\n\nSample of prior of program {}:'.format(i))
#         print(graph)
#
#         samples = mhgibbs(graph, int(1e4))
#         print(samples)
#         #process_results(stream, f"{i}.daphne", "graph_based_sampler", bins[i-1])
#
