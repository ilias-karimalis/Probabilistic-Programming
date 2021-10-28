import torch
import torch.distributions as tdist


# Implements a first class Distribution object which we pass around in our
# evaluator
class Distribution:
    def __init__(self, dist):
        self.dist = dist

    def sample(self):
        return self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)


class Normal(Distribution):
    def __init__(self, args):
        assert (len(args) == 2)
        self.mean = args[0]
        self.covariance = args[1]
        super().__init__(tdist.normal.Normal(self.mean, self.covariance))


class UniformContinuous(Distribution):
    def __init__(self, args):
        assert (len(args) == 2)
        self.lower = args[0]
        self.upper = args[1]
        super().__init__(tdist.uniform.Uniform(self.lower, self.upper))


class Beta(Distribution):
    def __init__(self, args):
        assert (len(args) == 2)
        self.alpha = args[0]
        self.beta = args[1]
        super().__init__(tdist.beta.Beta(self.alpha, self.beta))


class Exponential(Distribution):
    def __init__(self, args):
        assert (len(args) == 1)
        self.mu = args[0]
        super().__init__(tdist.exponential.Exponential(self.mu))


class Categorical(Distribution):
    def __init__(self, args):
        assert (len(args) == 1)
        self.probs = torch.flatten(args[0])
        super().__init__(tdist.categorical.Categorical(probs=self.probs))


class Gamma(Distribution):
    def __init__(self, args):
        assert (len(args) == 2)
        self.concentration = args[0]
        self.rate = args[1]
        super().__init__(tdist.gamma.Gamma(self.concentration, self.rate))


class Dirichlet(Distribution):
    def __init__(self, args):
        assert (len(args) == 1)
        self.concentration = torch.flatten(args[0])
        super().__init__(tdist.dirichlet.Dirichlet(self.concentration))


class Bernoulli(Distribution):
    def __init__(self, args):
        assert (len(args) == 1)
        self.p = args[0]
        super().__init__(tdist.bernoulli.Bernoulli(self.p))

# This seems kind of stupid but, I think we can treat the Dirac
# distribution as how it's mathematically defined (a limit of
# Normal distributions with var -> 0). We can thus define the Dirac
# as a distribution which always samples it's mean and has a very small
# variance (1e-2) <- This choice of variance is kind of random but,
# I'm unsure how to pick a better one...
class Dirac(Distribution):
    def __init__(self, args):
        assert (len(args) == 1)
        self.mean = args[0]
        self.covariance = torch.tensor(1e-2)
        super().__init__(tdist.normal.Normal(self.mean, self.covariance))

    def sample(self):
        return self.mean

