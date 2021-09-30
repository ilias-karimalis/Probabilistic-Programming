import numpy as np
from tqdm.auto import tqdm

# Q3

# first define the probability distributions as defined in the excercise:

# define 0 as false, 1 as true
FALSE = 0
TRUE = 1
SAMPLE_SPACE = [TRUE, FALSE]

def p_C(c):
    p = np.array([0.5, 0.5])
    return p[c]


def p_S_given_C(s, c):
    p = np.array([[0.5, 0.9], [0.5, 0.1]])
    return p[s, c]


def p_R_given_C(r, c):
    p = np.array([[0.8, 0.2], [0.2, 0.8]])
    return p[r, c]


def p_W_given_S_R(w, s, r):
    p = np.array([
        [[1.0, 0.1], [0.1, 0.01]],  # w = False
        [[0.0, 0.9], [0.9, 0.99]],  # w = True
    ])
    return p[w, s, r]

# Change DEBUG value to False to turn off logging
DEBUG = False
def log(s):
    if DEBUG:
        print(s)

# 1. enumeration and conditioning:

# compute joint:
p = np.zeros((2, 2, 2, 2))  # c,s,r,w
for c in range(2):
    for s in range(2):
        for r in range(2):
            for w in range(2):
                p[c, s, r, w] = p_C(c) * p_S_given_C(s, c) * p_R_given_C(r, c) * p_W_given_S_R(w, s, r)

# condition and marginalize:

# p(C=True)
prob_C = 0.5

# p(W=True)
prob_W = 0
for c in range(2):
    for s in range(2):
        for r in range(2):
            prob_W += p[c, s, r, 1]

# p(W=True | C=True)
prob_W_given_C = 0
for s in range(2):
    for r in range(2):
        prob_W_given_C += p_W_given_S_R(1, s, r) * p_R_given_C(r, 1) * p_S_given_C(s, 1)

prob_C_given_W = prob_W_given_C * prob_C / prob_W

print('There is a {:.2f}% chance it is cloudy given the grass is wet'.format(prob_C_given_W * 100))

# 2. ancestral sampling and rejection:
def sample_C():
    u = np.random.random()
    return TRUE if u <= p_C(TRUE) else FALSE


def sample_R_given_C(c):
    if c not in SAMPLE_SPACE:
        raise Exception("Invalid C value")
    u = np.random.random()
    return TRUE if u <= p_R_given_C(TRUE, c) else FALSE

def sample_S_given_C(c):
    if c not in SAMPLE_SPACE:
        raise Exception("Invalid C value")
    u = np.random.random()
    return TRUE if u <= p_S_given_C(TRUE, c) else FALSE

def sample_W_given_R_S(r,s):
    if r not in SAMPLE_SPACE:
        raise Exception("Invalid R value")
    if s not in SAMPLE_SPACE:
        raise Exception("Invalid S value")

    u = np.random.random()
    return TRUE if u <= p_W_given_S_R(TRUE, s, r) else FALSE

def ancestral_sample_C_S_R_W():
    c = sample_C()
    s = sample_S_given_C(c)
    r = sample_R_given_C(c)
    w = sample_W_given_R_S(r,s)
    return (c,s,r,w)


num_samples = 10000000
samples = np.zeros(num_samples)
rejections = 0
i = 0
while i < num_samples:
    # Sample (C,S,R,W) using ancestral sampling
    (c,s,r,w) = ancestral_sample_C_S_R_W()
    # Reject sample if W != True
    if w != 1:
        rejections += 1
        continue
    # C <- C_sample
    samples[i] = c
    i += 1

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean() * 100))
print('{:.2f}% of the total samples were rejected'.format(100 * rejections / (samples.shape[0] + rejections)))

# 3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:

# we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
# indexing is [c,s,r,w]
p_R_given_C_S_W = p / p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p / p.sum(axis=1, keepdims=True)

# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
# p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

# first create p(R,S,C):
p_C_S_R = np.zeros((2, 2, 2))  # c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c, s, r] = p_C(c) * p_S_given_C(s, c) * p_R_given_C(r, c)

# then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:, :, :] / p_C_S_R[:, :, :].sum(axis=(0), keepdims=True)

# gibbs sampling
num_samples = 100000000
samples = np.zeros(num_samples)
state = np.zeros(4, dtype='int')
# c,s,r,w, set w = True
(c,s,r,w) = (1,1,1,1)
for i in tqdm(range(num_samples)):
    log("*"*20)
    log("Iteration {}:".format(i+1))

    # sample S given C,R,W
    u2 = np.random.random()
    s = TRUE if u2 <= p_S_given_C_R_W[c,TRUE,r,w] else FALSE
    log("Value of S: {}".format(s))

    # sample R given C,S,W
    u3 = np.random.random()
    r = TRUE if u3 <= p_R_given_C_S_W[c,s,TRUE,w] else FALSE
    log("Value of R: {}".format(r))

    # sample C given S,R (Conditionally independent of W)
    u1 = np.random.random()
    c = TRUE if u1 <= p_C_given_S_R[TRUE,s,r] else FALSE
    log("Value of C: {}".format(c))

    # add new C value to samples
    samples[i] = c


print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean() * 100))
