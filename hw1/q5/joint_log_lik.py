import numpy as np
import scipy.special as sc

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
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
    #TODO
    (n_docs, n_topics) = doc_counts.shape
    (_, alphabet_size) = topic_counts.shape

    logBeta_alpha = - sum(sc.gammaln(alpha)) - sc.gammaln(sum(alpha))
    logBeta_gamma = - sum(sc.gammaln(gamma)) - sc.gammaln(sum(gamma))

    x = sum([sum(sc.gammaln(doc_counts[d, :] + alpha)) - sc.gammaln(sum(doc_counts[d, :] + alpha)) for d in range(n_docs)])
    y = sum([sum(sc.gammaln(topic_counts[k,:] + gamma)) - sc.gammaln(sum(topic_counts[k,:] + gamma)) for k in range(n_topics)])
    

    return logBeta_alpha * n_docs + logBeta_gamma * n_topics + x + y
    
