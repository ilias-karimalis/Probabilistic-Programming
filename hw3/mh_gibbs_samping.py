import torch
from tqdm.auto import tqdm
import time

from primitives import core
from utils import get_sampled, generate_markov_blankets, sample_from_priors, deterministic_eval


def mhgibbs_num_samples(graph, num_samples):

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


def mhgibbs_max_time(graph, max_time):

    nodes = graph[1]['V']
    link_functions = graph[1]['P']
    edges = graph[1]['A']
    observed = graph[1]['Y']

    print(link_functions)

    sampled = get_sampled(nodes, link_functions)
    markov_blankets = generate_markov_blankets(nodes, observed, edges)

    start = time.time()
    samples = [sample_from_priors(graph)]
    while True:
        if time.time() - start > max_time:
            break

        sample = mhgibbs_step(link_functions, sampled, observed, samples[-1], markov_blankets)
        samples.append(sample)

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