import time

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

        # Graph Processing
        self.nodes = self.graph[1]['V']
        self.link_functions = self.graph[1]['P']
        self.edges = self.graph[1]['A']
        self.sorted_nodes = topologicalSort(self.nodes, self.edges)
        self.observed = self.graph[1]['Y']
        self.sampled = self.__get_sampled()
        self.result_nodes = self.graph[2]
        self.markov_blankets = self.__generate_markov_blankets()

        # Misc
        self.optim = torch.optim.Adam([torch.tensor(0.0)], lr=self.learning_rate)


        # Initialize VI Distributions as Priors:
        self.vi_dists = {}
        local_env = self.observed
        for node in self.sorted_nodes:
            if node in self.sampled:
                dist = self.__deterministic_eval(self.link_functions[node][1], local_env)
                local_env[node] = dist.sample().detach()
                self.vi_dists[node] = dist.make_copy_with_grads()
                self.optim.param_groups[0]['params'] += self.vi_dists[node].Parameters()

        program_name = args["program_name"]
        wandb.init(project=f"BBVI for {program_name}", )
        wandb.config = args

    def __elbo_gradients(self, weights, weights_grad):
        g_hat = {}
        for node in self.sampled:
            for l in range(self.batch_size):
                if len(weights_grad[l][node]) == 1:
                    grad = weights_grad[l][node][0]
                else:
                    grad = torch.tensor(weights_grad[l][node])

                update = grad * weights[l] / self.batch_size
                g_hat[node] = (g_hat[node] + update) if node in g_hat else update

        return g_hat

        # F = [{}]*self.batch_size
        # g_hat = {v: [] for v in self.sampled}
        # node_set = set()
        # for wg in weights_grad:
        #     node_set.union(wg.keys())
        # for node in node_set:  # For each rv
        #     for l in range(self.batch_size):  # For each gradient itteration
        #         if node in weights_grad[l].keys():
        #             F[l][node] = {}
        #
        #             for i, grad in enumerate(weights_grad[l][node]):
        #                 F[l][node][i] = grad * weights[l]
        #         param_count = len(F[l][node].keys())
        #         param_shapes = {node: }
        #
        #
        #     b_hat = 1.0
        #     g_hat[node] = torch.tensor(sum([F[l][node] - b_hat*weights_grad[l][node] for l in range(self.batch_size)])) / self.batch_size
        return g_hat

    def __optimizer_step(self, g_hat):
        for dist in g_hat.keys():
            for i, parameter in enumerate(self.vi_dists[dist].Parameters()):
                parameter.grad = -g_hat[dist][i]
        self.optim.step()
        self.optim.zero_grad()

    def run(self):
        samples = []
        weights = []

        start = time.time()

        while time.time() - start < self.max_time:
            weights_batch = []
            weights_grad_batch = []
            for _ in range(self.batch_size):
                sigma = {'logW': torch.tensor(0.0), 'G': {}}
                res, sigma = self.__evaluate(sigma)
                samples.append(res)
                # print(f"Sigma: {sigma}")
                weights_batch.append(sigma["logW"])
                weights_grad_batch.append(sigma["G"])
                # self.optim.zero_grad()

            g_hat = self.__elbo_gradients(weights_batch, weights_grad_batch)
            self.__optimizer_step(g_hat)
            weights.extend(weights_batch)

            wandb.log({
                'ELBO': torch.mean(torch.tensor(weights_batch)).item(),
                'mean_mu': torch.mean(torch.tensor(samples)).item(),
                'mu': samples[-1]
            })

        wandb.finish()

        # weights = [w.detach() for w in weights]
        return samples

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
                sigma['logW'] += dist.log_prob(c)
                local_env[node] = c.detach()

            else:
                print("peepeepoopoo we have a bug ...")

        # print("LogW: {}".format(sigma["logW"]))
        # print("grad: {}".format(sigma["G"]))
        return self.__deterministic_eval(self.result_nodes, local_env), sigma

    # def __probabilistic_eval(self, node, local_env, sigma):
    #     op, *args = self.link_functions[node]
    #     if op == 'sample*':
    #         dist = self.__deterministic_eval(args[0], local_env)
    #         if node not in sigma['Q'].keys():
    #             sigma['Q'][node] = dist.make_copy_with_grads()  # MAYBE WRONG\
    #             self.optim.param_groups[0]['params'] += sigma['Q'][node].Parameters()
    #         c = sigma['Q'][node].sample()
    #         q_log_prob = sigma['Q'][node].log_prob(c)
    #         q_log_prob.backward()
    #         sigma['G'][node] = [param.grad.clone().detach() for param in sigma['Q'][node].Parameters()]
    #         sigma['logW'] += dist.log_prob(c) - q_log_prob
    #         return c, sigma
    #
    #     if op == 'observe*':
    #         dist = self.__deterministic_eval(args[0], local_env)
    #         val = self.__deterministic_eval(args[1], local_env)
    #         sigma['logW'] += dist.log_prob(val).detach()
    #         return val, sigma
    #
    #     print("peepeepoopoo")

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

    def __generate_markov_blankets(self):
        markov_blankets = {}
        for node in self.sampled:
            node_blanket = [node]
            node_blanket.extend(self.edges[node])
            markov_blankets[node] = node_blanket
        return markov_blankets

