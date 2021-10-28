import time

import torch
import torch.distributions as tdist
from daphne import daphne

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
        new_sample = sample.detach() - self.epsilon * momentum
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
        # u = torch.tensor(0.0)
        # for node in self.nodes:
        #
        # return torch.sum(
        #     torch.tensor([- deterministic_eval(self.link_functions[node][1], env).log_prob(env[node]) for node in self.nodes])
        # )
        return - self.__log_joint(env)

    def __nabla_u(self, sample):
        sample.requires_grad_(True)
        u = self.__u(sample)
        u.backward()
        return sample.grad

    def __log_joint(self, env):
        ret = 0
        for node in self.nodes:
            dist = deterministic_eval(self.link_functions[node][1], env)
            val = env[node]
            if type(val) in [int, float, bool]:
                val = torch.tensor(float(val))
            ret += dist.log_prob(val)

        return ret



# # def hmc(graph, num_samples, num_leaps, epsilon, m_matrix):
# def hmc_num_samples(args):
#     # Parse Arguments
#     graph = args["graph"]
#     num_samples = args["num_samples"]
#     num_leaps = args["num_leaps"]
#     epsilon = args["epsilon"]
#     m_matrix = args["m_matrix"]
#
#     # Get Graph Stuff
#     nodes = graph[1]['V']
#     link_functions = graph[1]['P']
#     edges = graph[1]['A']
#     observed = graph[1]['Y']
#     result_nodes = graph[2]
#     sampled = get_sampled(nodes, link_functions)
#
#     # Misc
#     samples = []
#     udist = tdist.uniform.Uniform(0, 1)
#     normal = tdist.normal.Normal(0, m_matrix)
#
#     # Setup initial env
#     prior_samples = sample_from_priors(graph)
#     env = {key: enable_gradient(value) for key, value in prior_samples if key in sampled}
#
#     for s in range(num_samples):
#         momentum = normal.sample()
#         env_last = {key: enable_gradient(value) for key, value in env}
#         env_new, momentum_new = hmc_leapfrog(env_last, link_functions, momentum, num_leaps, epsilon)
#         acceptance_rate = hmc_acceptance(env_last, env_new, momentum, momentum_new, link_functions, observed, m_matrix)
#         u = udist.sample()
#         if u < acceptance_rate:
#             env = env_new
#         samples.append(({**env, **observed}, deterministic_eval(result_nodes, {**env, **observed})))
#     return samples
#
#
# def hmc_max_time(args):
#     # Parse Arguments
#     graph = args["graph"]
#     max_time = args["max_time"]
#     num_leaps = args["num_leaps"]
#     epsilon = args["epsilon"]
#     m_value = args["m_value"]
#
#     # Get Graph Stuff
#     nodes = graph[1]['V']
#     link_functions = graph[1]['P']
#     observed = {key: torch.tensor(float(value)) for key, value in graph[1]['Y'].items()}
#     result_nodes = graph[2]
#     sampled = get_sampled(nodes, link_functions)
#
#     # Misc
#     samples = []
#     udist = tdist.uniform.Uniform(0, 1)
#     m_matrix = torch.diag(m_value * torch.ones(len(sampled)))
#     zeros_vec = torch.zeros(len(sampled))
#     normal = tdist.normal.Normal(zeros_vec, m_matrix)
#
#     # Setup initial env
#     prior_samples = sample_from_priors(graph)
#     env = {key: enable_gradient(value) for key, value in prior_samples.items() if key in sampled}
#
#     start = time.time()
#     while True:
#         if time.time() - start > max_time:
#             break
#         momentum = normal.sample()
#         env_last = {key: enable_gradient(value) for key, value in env.items()}
#         env_new, momentum_new = hmc_leapfrog(env_last, link_functions, momentum, num_leaps, epsilon)
#         acceptance_rate = hmc_acceptance(env_last, env_new, momentum, momentum_new, link_functions, observed, m_matrix)
#         u = udist.sample()
#         if u < acceptance_rate:
#             env = env_new
#         samples.append(({**env, **observed}, deterministic_eval(result_nodes, {**env, **observed})))
#     return samples
#
#
#
# def hmc_leapfrog(env, link_functions, momentum0, num_leaps, epsilon):
#     u = hmc_u(env, link_functions)
#     u.backward()
#     momentum = momentum0 - 0.5 * epsilon * hmc_nabla_u(env)
#     for _ in range(num_leaps - 1):
#         env, momentum = hmc_leapfrog_step(env, momentum, epsilon)
#
#     # Last half step:
#     return hmc_leapfrog_step(env, momentum, epsilon, half_step=True)
#
#
# def hmc_leapfrog_step(env, momentum, epsilon, half_step=False):
#     take_step = lambda i, value: enable_gradient(value + epsilon * momentum[i])
#     new_env = {key: take_step(i, value) for i, (key, value) in enumerate(env.items())}
#     if half_step:
#         new_momentum = momentum - 0.5 * epsilon * hmc_nabla_u(new_env)
#     else:
#         new_momentum = momentum - epsilon * hmc_nabla_u(new_env)
#     return new_env, new_momentum
#
#
# # def hmc_update_dict(env, function, cond=lambda i, k, v: True):
# #     return { key: function(i, key, value) for i, (key, value) in enumerate(env.items()) if cond(i, key, value)}
#
#
# def hmc_acceptance(env_last, env_new, momentum_last, momentum_new, link_functions, observed, m_mat):
#     return torch.exp(
#         - hmc_hamiltonian({**env_new, **observed}, momentum_last, link_functions, m_mat)
#         + hmc_hamiltonian({**env_last, **observed}, momentum_new, link_functions, m_mat)
#     )
#
#
# def enable_gradient(tensor):
#     tensor = tensor.detach()
#     tensor.requires_grad = True
#     return tensor
#
#
# def hmc_u(env, link_functions):
#     evaluated_trajectories = [deterministic_eval(link_functions[key][1], {**core, **env}).log_prob(env[key]) for key in env.keys()]
#     return - torch.sum(torch.tensor(evaluated_trajectories, requires_grad=True))
#
#
# def hmc_hamiltonian(env, momentum, link_functions, m_matrix):
#     u = hmc_u(env, link_functions)
#     k = 0.5 * torch.matmul(momentum.T, torch.matmul(m_matrix, momentum))
#     return u + k
#
#
# def hmc_nabla_u(env):
#     return torch.tensor([value.grad for value in env.values()])


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


if __name__ == '__main__':

    for i in range(1, 5):
        graph = daphne(['graph', '-i', '../hw3/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
