import torch
import torch.distributions as tdist
from daphne import daphne

from primitives import core
from utils import topologicalSort


def hmc(graph, num_samples, num_leaps, epsilon, m):
    nodes = graph[1]['V']
    link_functions = graph[1]['P']
    edges = graph[1]['A']
    observed = graph[1]['Y']

    sorted_nodes = topologicalSort(nodes, edges)

    samples = [sample_from_priors(graph)]
    normal = tdist.normal.Normal(0, m)
    for s in range(num_samples):
        momentum = normal.sample()
        last_sample = samples[s]

        new_sample, new_momentum = leapfrog(sorted_nodes, last_sample, momentum, num_leaps, epsilon)

        u = torch.rand(1)
        acceptance_rate = torch.exp(- 1.0 * hamiltonian(graph, new_sample, new_momentum)
                                    + hamiltonian(graph, last_sample, momentum))
        if u < acceptance_rate:
            samples.append(new_sample)
        else:
            samples.append(last_sample)


def potential(graph, sampled, observed):
    pass

def hamiltonian(graph, sample, momentum, m):
    u = potential(sample)
    k = 0.5 * torch.matmul(momentum.T, torch.matmul(m, momentum))
    return u + k


def leapfrog(sorted_nodes, last_sample, momentum_0, num_leaps, epsilon):
    momentum = momentum_0 - 0.5 * epsilon * nabla_u(sorted_nodes, last_sample)
    sample = last_sample
    for _ in range(num_leaps - 1):
        sample = sample + epsilon * momentum
        momentum = momentum - epsilon * nabla_u(sorted_nodes, sample)

    sample = sample + epsilon * momentum
    momentum = momentum_0 - 0.5 * epsilon * nabla_u(sorted_nodes, sample)
    return sample, momentum


def nabla_u(sorted_nodes, sample):
    joint_log_prob = 0

    for node in sorted_nodes:
        return 1


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


def probabilistic_eval(exp, env):
    op, *args = exp
    if op == 'sample*':
        dist = deterministic_eval(args[0], env)
        return dist.sample()
    elif op == 'observe*':
        _, observed = args
        return deterministic_eval(observed, env)


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


if __name__ == '__main__':


    for i in range(1,5):
        graph = daphne(['graph','-i','../hw3/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
