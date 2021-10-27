import torch
from tqdm.auto import tqdm

from primitives import core
from toposort import topologicalSort


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


def mhgibbs(graph, num_samples):

    nodes = graph[1]['V']
    link_functions = graph[1]['P']
    edges = graph[1]['A']
    observed = graph[1]['Y']

    sampled = get_sampled(nodes, link_functions)
    markov_blankets = generate_markov_blankets(nodes, observed, edges)

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