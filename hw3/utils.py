import numpy as np
import torch
from primitives import core


def weighted_avg_and_var(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance


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


def generate_markov_blankets(nodes, observed, edges):

    markov_blankets = {}

    for node in nodes:
        if node not in observed:
            node_blanket = [node]
            node_blanket.extend(edges[node])
            markov_blankets[node] = node_blanket

    return markov_blankets


def get_sampled(nodes, link_functions):
    sampled = []
    for node in nodes:
        if link_functions[node][0] == "sample*":
            sampled.append(node)

    return sampled

