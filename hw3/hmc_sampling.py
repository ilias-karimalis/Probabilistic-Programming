import time

import torch
import torch.distributions as tdist

from primitives import core
from utils import topologicalSort, get_sampled


class HMCSampler:

    def __init__(self, args):
        # Parse args
        self.graph = args["graph"]
        self.max_time = args["max_time"]
        self.num_leaps = args["num_leaps"]
        self.epsilon = args["epsilon"]

        # Graph Processing
        self.nodes = self.graph[1]['V']
        self.link_functions = self.graph[1]['P']
        self.observed = {key: torch.tensor(float(value)) for key, value in self.graph[1]['Y'].items()}
        self.result_nodes = self.graph[2]
        self.sampled = get_sampled(self.nodes, self.link_functions)

    def run(self):
        zero_vec = torch.zeros(len(self.sampled))
        m_matrix = torch.eye(len(self.sampled))
        momentum_dist = tdist.multivariate_normal.MultivariateNormal(zero_vec, m_matrix)
        u_dist = tdist.uniform.Uniform(0, 1)

        samples = []
        log_joints = []

        # Start Timer
        start = time.time()

        prior_samples = sample_from_priors(self.graph)
        sample = torch.tensor([prior_samples[key] for key in prior_samples if key in self.sampled])

        while time.time() - start < self.max_time:
            momentum = momentum_dist.sample()
            sample_new, momentum_new = self.__leapfrog(sample, momentum)
            u = u_dist.sample()
            acceptance_rate = self.__acceptance_rate(sample, sample_new, momentum, momentum_new, m_matrix)

            if u < acceptance_rate:
                env = {**dict(zip(self.sampled, sample_new)), **self.observed}
                samples.append(deterministic_eval(self.result_nodes, env).detach())
                log_joints.append(self.__log_joint(env).detach())
                sample = sample_new

            else:
                env = {**dict(zip(self.sampled, sample)), **self.observed}
                samples.append(deterministic_eval(self.result_nodes, env).detach())
                log_joints.append(self.__log_joint(env).detach())

        return samples, log_joints

    def __leapfrog(self, sample, momentum):
        sample = sample.detach().clone()
        momentum = momentum - 0.5 * self.epsilon * self.__nabla_u(sample)
        for _ in range(self.num_leaps - 1):
            # sample = sample.detach() - self.epsilon * momentum
            # momentum = momentum - self.epsilon * self.__nabla_u(sample)
            sample, momentum = self.__leapfrog_step(sample, momentum)

        # sample = sample.detach() + self.epsilon * momentum
        # momentum = momentum - 0.5 * self.epsilon * self.__nabla_u(sample)
        # return sample, momentum
        return self.__leapfrog_step(sample, momentum, half_step=True)

    def __leapfrog_step(self, sample, momentum, half_step=False):
        new_sample = sample.detach() + self.epsilon * momentum
        if half_step:
            new_momentum = momentum - 0.5 * self.epsilon * self.__nabla_u(new_sample)
        else:
            new_momentum = momentum - self.epsilon * self.__nabla_u(new_sample)
        return new_sample, new_momentum

    def __acceptance_rate(self, sample, sample_new, momentum, momentum_new, m_matrix):
        return torch.exp(
            - self.__hamiltonian(sample_new, momentum_new, m_matrix)
            + self.__hamiltonian(sample, momentum, m_matrix)
        )

    def __hamiltonian(self, sample, momentum, m_matrix):
        sample.requires_grad_(True)
        u = self.__u(sample)
        k = 0.5 * torch.matmul(momentum.T, torch.matmul(m_matrix.inverse(), momentum))
        return u + k

    def __u(self, sample):
        env = {**dict(zip(self.sampled, sample)), **self.observed}
        return - self.__log_joint(env)

    def __nabla_u(self, sample):
        sample.requires_grad_(True)
        u = self.__u(sample)
        u.backward()
        return sample.grad

    def __log_joint(self, env):
        ret = torch.tensor(0.0)
        for node in self.nodes:
            dist = deterministic_eval(self.link_functions[node][1], env)
            val = env[node]
            if type(val) in [int, float, bool]:
                val = torch.tensor(float(val))
            ret += dist.log_prob(val)

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
    proc = core[op]
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
