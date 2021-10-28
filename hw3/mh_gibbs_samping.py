import torch
import torch.distributions as tdist
import time

from primitives import core
from utils import get_sampled, sample_from_priors, deterministic_eval


class MHGibbsSampler:

    def __init__(self, args):
        # Parse args
        self.graph = args["graph"]
        self.max_time = args["max_time"]

        # Graph Processing
        self.nodes = self.graph[1]['V']
        self.link_functions = self.graph[1]['P']
        self.edges = self.graph[1]['A']
        self.observed = self.graph[1]['Y']
        self.result_nodes = self.graph[2]
        self.sampled = get_sampled(self.nodes, self.link_functions)
        self.markov_blankets = self.__generate_markov_blankets()

        # Misc
        self.u_dist = tdist.uniform.Uniform(0, 1)

    def run(self):
        samples = []
        log_joints = []

        start = time.time()

        samples.append(sample_from_priors(self.graph))

        while time.time() - start < self.max_time:
            sample = self.__step(samples[-1])
            samples.append(sample)
            log_joints.append(self.__log_joint({**sample, **self.observed, **core}))

        result_samples = [deterministic_eval(self.result_nodes, {**sample, **self.observed, **core}) for sample in samples]
        return result_samples, log_joints

    def __step(self, last_sample):
        sample = last_sample.copy()
        for rv in self.sampled:
            # Draw Samples from Prior for proposal
            dist = deterministic_eval(self.link_functions[rv][1], {**core, **last_sample})
            proposed_sample = sample.copy()
            proposed_sample[rv] = dist.sample()
            u = self.u_dist.sample()
            alpha = self.__acceptance(rv, proposed_sample, sample)
            if u < alpha:
                sample = proposed_sample
        return sample

    def __acceptance(self, rv, proposed_sample, last_sample):
        dist_new = deterministic_eval(self.link_functions[rv][1], {**core, **proposed_sample})
        dist_last = deterministic_eval(self.link_functions[rv][1], {**core, **last_sample})
        log_alpha = dist_new.log_prob(last_sample[rv]) - dist_last.log_prob(proposed_sample[rv])
        for node in self.markov_blankets[rv]:
            log_alpha += deterministic_eval(
                self.link_functions[node][1],
                {**core, **self.observed, **proposed_sample}
            ).log_prob(proposed_sample[node])
            log_alpha -= deterministic_eval(
                self.link_functions[node][1],
                {**core, **self.observed, **last_sample}
            ).log_prob(last_sample[node])

        return torch.exp(log_alpha)

    def __generate_markov_blankets(self):
        markov_blankets = {}

        for node in self.nodes:
            if node not in self.observed:
                node_blanket = [node]
                node_blanket.extend(self.edges[node])
                markov_blankets[node] = node_blanket

        return markov_blankets

    def __log_joint(self, env):
        ret = torch.tensor(0.0)
        for node in self.nodes:
            dist = deterministic_eval(self.link_functions[node][1], env)
            val = env[node]
            if type(val) in [int, float, bool]:
                val = torch.tensor(float(val))
            ret += dist.log_prob(val)

        return ret
