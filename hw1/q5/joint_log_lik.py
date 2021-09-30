import numpy as np
import scipy.special as sc

def initialize_jll_constants(alpha, gamma, n_docs):
    logBeta_alpha = np.sum(sc.gammaln(alpha)) - sc.gammaln(np.sum(alpha))
    logBeta_gamma = n_docs * sc.gammaln(gamma) - sc.gammaln(n_docs * gamma)
    return logBeta_alpha, logBeta_gamma

def joint_log_lik(doc_counts, topic_counts, alpha, gamma, logBeta_alpha, logBeta_gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        jll: the joint log likelihood of the model
    """
    (n_docs, n_topics) = doc_counts.shape
    x = np.sum([np.sum(sc.gammaln(doc_counts[d, :] + alpha)) - sc.gammaln(np.sum(doc_counts[d, :] + alpha)) for d in range(n_docs)])
    y = np.sum([np.sum(sc.gammaln(topic_counts[k,:] + gamma)) - sc.gammaln(np.sum(topic_counts[k,:] + gamma)) for k in range(n_topics)])

    return - logBeta_alpha * n_docs - logBeta_gamma * n_topics + x + y
