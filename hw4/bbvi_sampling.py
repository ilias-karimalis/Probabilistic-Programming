import time

import numpy as np
import torch
import wandb

from utils import topologicalSort
from primitives import core


class BBVISampler:

    def __init__(self, args):
        # Parse args
        self.graph = args['graph']
        self.max_time = args['max_time']
        self.learning_rate = args['learning_rate']
        self.batch_size = args['batch_size']
        self.use_wandb = args["wandb?"]

        # Initialize Optimizer
        self.optim = torch.optim.Adam([torch.tensor(0.0)], lr=self.learning_rate)
        if "optimizer" in args.keys():
            self.optim = args["optimizer"]([torch.tensor(0.0)], lr=self.learning_rate)

        # Graph Processing
        self.nodes = self.graph[1]['V']
        self.link_functions = self.graph[1]['P']
        self.edges = self.graph[1]['A']
        self.sorted_nodes = topologicalSort(self.nodes, self.edges)
        self.observed = self.graph[1]['Y']
        self.sampled = self.__get_sampled()
        self.result_nodes = self.graph[2]

        # Initialize VI Distributions as Priors:
        self.vi_dists = {}
        local_env = self.observed
        for node in self.sorted_nodes:
            if node in self.sampled:
                dist = self.__deterministic_eval(self.link_functions[node][1], local_env)
                local_env[node] = dist.sample().detach()
                self.vi_dists[node] = dist.make_copy_with_grads()
                self.optim.param_groups[0]['params'] += self.vi_dists[node].Parameters()

        if self.use_wandb:
            program_name = args["program_name"]
            wandb.init(project=f"BBVI for {program_name}")

    def run(self):
        samples = []
        weights = []

        Q_max = self.vi_dists.copy()
        ELBO_max = -torch.inf

        start = time.time()

        while time.time() - start < self.max_time:
            weights_batch = []
            samples_batch = []
            weights_grad_batch = []
            for _ in range(self.batch_size):
                sigma = {'logW': torch.tensor(0.0), 'G': {}}
                res, sigma = self.__evaluate(sigma)

                samples_batch.append(res)
                weights_batch.append(sigma["logW"])
                weights_grad_batch.append(sigma["G"])
                self.optim.zero_grad()

            # TODO: Not sure if this is needed anymore but, this is useful in avoiding
            # exploding ELBO which seemed to happen sometimes...
            elbo = torch.mean(torch.tensor(weights_batch)).item()
            if elbo < -1e8:
                continue

            g_hat = self.__elbo_gradients(weights_batch, weights_grad_batch)
            self.__optimizer_step(g_hat)
            weights.extend(weights_batch)
            samples.extend(samples_batch)

            elbo = torch.mean(torch.tensor(weights_batch)).item()
            if elbo > ELBO_max:
                Q_max = self.vi_dists.copy()
            if self.use_wandb:
                wandb.log({
                    'ELBO': elbo,
                })
        if self.use_wandb:
            wandb.finish()

        return samples, weights, Q_max

    def __elbo_gradients(self, weights, weights_grad):
        g_hat = {v: [] for v in self.sampled}

        for node in self.sampled:  # Over rv
            num_params = len(weights_grad[0][node])
            node_grad = [weights_grad[i][node] for i in range(self.batch_size)]
            g_hat[node] = self.__node_gradient(node_grad, weights, num_params)

        return g_hat

    def __node_gradient(self, node_grad, weights, num_params):
        G_dict, F_dict = self.__generate_FG_dicts(node_grad, num_params, weights)
        param_lengths = [G_dict[p].size()[1] for p in range(num_params)]
        g_hat_node = []

        for p in range(num_params):
            b_hat = torch.zeros(param_lengths[p])
            cov_sum = torch.tensor(0.0)
            var_sum = torch.tensor(0.0)
            for p_val in range(param_lengths[p]):
                cov_matrix = torch.cov(torch.stack((G_dict[p][:, p_val], F_dict[p][:, p_val]), dim=0))
                cov_p, var_p = cov_matrix[0, 1], cov_matrix[1, 1]
                b_hat[p_val] = cov_p / var_p
                cov_sum += cov_p
                var_sum += var_p
            b_hat_p = cov_sum / var_sum
            g_hat_node_p = torch.sum(F_dict[p] - b_hat_p * G_dict[p])
            g_hat_node.append(g_hat_node_p.squeeze(0 if param_lengths[p] == 1 else g_hat_node_p))

        return g_hat_node

    # THIS FEELS SO SHIT HOW DO I DO THIS BETTER????
    def __generate_FG_dicts(self, node_grad, num_params, weights):
        G_dict = {d: [] for d in range(num_params)}
        F_dict = {d: [] for d in range(num_params)}
        for itter in range(self.batch_size):
            for d, dist_param in enumerate(node_grad[itter]):
                G_dict[d].append(dist_param)
                F_dict[d].append(dist_param * weights[itter])

        param_ndim = [G_dict[0][d].dim() for d in range(num_params)]

        # Now we turn each of these into Tensor elements to allow us to compute b_hat:
        for d in range(num_params):
            if param_ndim[d] > 0:
                G_dict[d] = torch.stack(G_dict[d], dim=0)
                F_dict[d] = torch.stack(F_dict[d], dim=0)
            else:
                G_dict[d] = torch.tensor(G_dict[d]).unsqueeze(1)
                F_dict[d] = torch.tensor(F_dict[d]).unsqueeze(1)

        return G_dict, F_dict

    def __optimizer_step(self, g_hat):
        for dist in g_hat.keys():
            for i, parameter in enumerate(self.vi_dists[dist].Parameters()):
                parameter.grad = -g_hat[dist][i]
        self.optim.step()
        self.optim.zero_grad()

    def __evaluate(self, sigma):
        local_env = {}
        for node in self.sorted_nodes:
            op, *args = self.link_functions[node]
            dist = self.__deterministic_eval(args[0], local_env)
            if op == 'sample*':
                c = self.vi_dists[node].sample()
                q_log_prob = self.vi_dists[node].log_prob(c)
                q_log_prob.backward()
                sigma['G'][node] = [param.grad.clone().detach() for param in self.vi_dists[node].Parameters()]
                sigma['logW'] += dist.log_prob(c).detach() - q_log_prob.detach()
                local_env[node] = c.detach()

            elif op == 'observe*':
                c = torch.tensor(self.observed[node])
                local_env[node] = c.detach()
                sigma['logW'] += dist.log_prob(c).detach()

        return self.__deterministic_eval(self.result_nodes, local_env), sigma

    def __deterministic_eval(self, exp, local_env):
        if type(exp) in [int, float, bool]:
            return torch.tensor(float(exp))

        elif type(exp) == str:
            if exp in local_env.keys():
                return local_env[exp]
            return core[exp]

        elif exp is None:
            return 0

        op, *args = exp
        if op == 'if':
            test, conseq, alt = args
            res_conseq = self.__deterministic_eval(conseq, local_env)
            res_alt = self.__deterministic_eval(alt, local_env)
            b = self.__deterministic_eval(test, local_env)
            return res_conseq if b else res_alt

        # Else call procedure:
        proc = self.__deterministic_eval(op, local_env)
        c = [0] * len(args)
        for (i, arg) in enumerate(args):
            c[i] = self.__deterministic_eval(arg, local_env)
        return proc(c)

    def __get_sampled(self):
        sampled = []
        for node in self.nodes:
            if self.link_functions[node][0] == "sample*":
                sampled.append(node)
        return sampled
