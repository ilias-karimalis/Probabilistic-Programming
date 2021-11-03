from importance_sampling import LWSampler

import torch

class BBVISampler:

    def __init__(self, args):
        # Parse args
        self.graph = args['graph']
        self.max_time = args['max_time']

        # Graph Processing
        self.nodes = self.graph[1]['V']
        self.link_functions = self.graph[1]['P']
        self.edges = self.graph[1]['A']
        self.observed = self.graph[1]['Y']
        self.result_noeds = self.graph[2]
        self.markov_blankets = self.__generate_markov_blankets()

    def __generate_markov_blankets(self):
        pass


# class BBVISampler(LWSampler):
#
#     def __init__(self, args):
#         super().__init__(args)
#
#     def optimizer_step(self, Q, g_hat):
#         lambda_dict = {}
#         for v in g_hat.keys():
#             lambda_dict[v] =  0#get_parameters(Q[v])
#
#     def run(self, T, L):
#         sigma = {
#             'logW': 0,
#             'Q': [],
#             'G': [],
#         }
#         for t in range(T):
#             for l in range(L):
#
#
#     def __eval(self, exp, sigma, env):
#         if type(exp) in [int, float, bool]:
#             return torch.tensor(float(exp)), sigma
#
#         elif type(exp) == str:
#             if exp in self.user_functions:
#                 return self.user_functions[exp], sigma
#             else:
#                 return env[exp], sigma
#
#         op, *args = exp
#         if op == 'sample':
#             dist, sigma = self.__eval(exp[1], sigma, env)
#             ident = id(exp[1])
#             if ident not in sigma["Q"].keys():
#                 sigma["Q"][ident] = dist
#             proposal = sigma["Q"][ident]
#             sample = proposal.sample()
#             sigma["G"][ident] = proposal.grad_log_prob(sample)
#             sigma['logW'] += dist.log_prob(sample) - proposal.log_prob(sample)
#             return sample, sigma
#
#         elif op == 'observe':
#             dist, sigma = self.__eval(exp[1], sigma, env)
#             observed_value, sigma = self.__eval(exp[2], sigma, env)
#             if type(observed_value) is bool:
#                 observed_value = torch.tensor(1.0 if observed_value else 0.0)
#             sigma['logW'] += dist.log_prob(observed_value)
#             return observed_value, sigma
#
#         elif op == 'let':
#             value, sigma = self.__eval(args[0][1], sigma, env)
#             env[args[0][0]] = value
#             return self.__eval(args[1], sigma, env)
#
#         elif op == 'if':
#             test, conseq, alt = args
#             b, sigma = self.__eval(test, sigma, env)
#             return self.__eval(conseq if b else alt, sigma, env)
#
#         proc, sigma = self.__eval(op, sigma, env)
#         c = [0] * len(args)
#         for (i, arg) in enumerate(args):
#             c[i], sigma = self.__eval(arg, sigma, env)
#         return proc(c, sigma, self.__eval) if proc in self.user_functions.values() else (proc(c), sigma)

