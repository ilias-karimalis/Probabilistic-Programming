from numpy.core.defchararray import join
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from joint_log_lik import initialize_jll_constants, joint_log_lik
from sample_topic_assignment import sample_topic_assignment
from tqdm.auto import tqdm
from datetime import datetime
import os

bagofwords = loadmat('bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  # go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('words_nips.mat')['WO'][:,0]
titles = loadmat('titles_nips.mat')['titles'][:,0]

# For Debugging
DEBUG=True
def log(s):
    if DEBUG:
        print(s)

def log_separator():
    log('*'*80)

# This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

# ***************************
# Initialize Hyperparameters:
# ***************************
# Wether to train on all or part of the data
mini_run = False
# number of topics
n_topics = 20
# prior parameters, alpha parameterizes the dirichlet to regularize the
# document specific distributions over topics and gamma parameterizes the
# dirichlet to regularize the topic specific distributions over words.
# These parameters are both scalars and really we use alpha * ones() to
# parameterize each dirichlet distribution. Iters will set the number of
# times your sampler will iterate.
alpha = np.ones(n_topics) * 0.5
gamma = 0.001
iters = 1000

# subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
# PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
if mini_run:
    words = words[document_assignment < 100]
    document_assignment  = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

# initial topic assigments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

# within document count of topics
doc_counts = np.zeros((n_docs,n_topics))

for d in range(n_docs):
    # histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5,n_topics-0.5))[0]

# doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

# within topic count of words
topic_counts = np.zeros((n_topics,alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5,alphabet_size-0.5))[0]


# topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)


hyperparmeter_string = """Number of Topics: {}
Number of Iterations: {}
Alpha: {}
Gamma: {}
Run Size: {}""".format(n_topics, iters, alpha[0], gamma, "Small Run" if mini_run else "Full Run")

logBeta_alpha, logBeta_gamma = initialize_jll_constants(alpha, gamma, n_docs)

# Setup Results folder for this run:
date_time_string = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
result_dir = "results/" + date_time_string + "/"
os.mkdir(result_dir)

# Add Hyperparameter file to the results directory
with open(result_dir + "hyperparameters.txt", 'w') as f:
    f.write(hyperparmeter_string)

jll = []
max_jll = (-1, -np.inf, None, None, None, None) # (argmax_jll, max_jll, max_topic_assignment, max_topic_counts, max_doc_counts, max_topic_N)
for i in tqdm(range(iters)):
    
    prm = np.random.permutation(words.shape[0])
    
    words = words[prm]   
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]
    
    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
                                topic_assignment,
                                topic_counts,
                                doc_counts,
                                topic_N,
                                doc_N,
                                alpha,
                                gamma,
                                words,
                                document_assignment)

    curr_jll = joint_log_lik(doc_counts,topic_counts,alpha,gamma,logBeta_alpha, logBeta_gamma) 
    jll.append(curr_jll)

    if curr_jll > max_jll[1]:
        max_jll = (i, curr_jll, topic_assignment, topic_counts, doc_counts, topic_N)


plt.plot(jll)
plt.savefig(result_dir + "full_plot.pdf")
plt.show()

plt.plot(range(99, iters), jll[99:])
plt.savefig(result_dir + "burnin_removal_plot.pdf")
plt.show()

#
# For the following questions we proceed by using the single sample that obtained the highest JLL
#
topic_assignment = max_jll[2]
topic_counts = max_jll[3]
doc_counts = max_jll[4]
topic_N = max_jll[5]

# find the 10 most probable words of the 20 topics:
fstr = ''
for topic in range(n_topics):
    words = np.argsort(topic_counts[topic,:])[-10:]
    fstr += "Topic {}:\n".format(topic)
    for word in words:
        fstr += WO[word][0]
        fstr += "\n"

with open(result_dir + 'most_probable_words_per_topic','w') as f:
    f.write(fstr)


# most similar documents to document 0 by cosine similarity over topic distribution:
# normalize topics per document and dot product:
csarray = [doc_counts[0].dot(doc_counts[doc])/(np.linalg.norm(doc_counts[0])*np.linalg.norm(doc_counts[doc])) for doc in range(n_docs)]
top_docs = np.argsort(csarray)[-11:]
fstr = "Top Cosine Similarities with Document 0\n"
for doc in reversed(top_docs):
    #print(doc_counts[doc,:].dot(doc_counts[doc,:]))
    fstr += "Document {}, Title: {}\n".format(doc, titles[doc])
    fstr += str(doc_counts[0].dot(doc_counts[doc])/(np.linalg.norm(doc_counts[0])*np.linalg.norm(doc_counts[doc]))) + "\n"

with open(result_dir + 'most_similar_titles_to_0','w') as f:
    f.write(fstr)
