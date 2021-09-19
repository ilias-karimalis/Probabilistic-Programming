from os import defpath
import numpy as np
from tqdm.auto import tqdm


# For Debugging
DEBUG=True
def log(s):
    if DEBUG:
        print(s)

def log_separator():
    log('*'*80)

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    # TODO
    (n,) = topic_assignment.shape
    (n_topics, alphabet_size) = topic_counts.shape
    (n_docs,_) = doc_counts.shape

    for i in tqdm(range(n)):
        word = words[i]
        topic = topic_assignment[i]
        document = document_assignment[i]

        # We must lower the topic values for this topic because We are conditioning on all topics but it,
        doc_counts[document, topic] -= 1
        topic_counts[topic, word] -= 1
        topic_N[topic] -= 1

        #p_z = np.zeros(n_topics)

        #for k in range(n_topics):
        #    x = (topic_counts[k, word] + gamma)
        #    y = (doc_counts[document, k] + alpha)
        #    w = (sum(topic_counts[k]) + alphabet_size*gamma)
        #    z = (doc_N[document] + n_topics*alpha)
        #    #print(doc_N[document])
        #    #print("x: {} \ny: {}\nw: {} \nz: {}".format(x,y,w,z))
        #    p_z[k] = x * y /( w * z)
        #     #(doc_counts[document, k] + alpha[k])*(topic_counts[k, word] + gamma[word])/(topic_N[k] + sum(gamma))

        #p_z = (topic_counts[:, word] + gamma) * (doc_counts[document, :] + alpha) / ((np.sum(topic_counts, axis=0) + alphabet_size*gamma) * (doc_N[document] + n_topics*alpha))
        x = (topic_counts[:, word] + gamma[word]) 
        y = (doc_counts[document, :] + alpha)
        w = (np.sum(topic_counts, axis=1) + alphabet_size*gamma[word])
        z = (doc_N[document] + n_topics*alpha) 
        #print("x: {} \ny: {}\nw: {} \nz: {}".format(x,y,w,z))
        p_z = (x * y)/(w * z)
        p_z = p_z / np.sum(p_z)

        #log(p_z)
        #log(sum(p_z))
        sample = np.argmax(np.random.multinomial(1, p_z))

        #print(sample)

        topic_assignment[i] = sample
        doc_counts[document, sample] += 1
        topic_counts[sample, word] += 1
        topic_N[sample] += 1


        
    #for in range()
    #log_separator()
    #log("Topic Assignment:")
    #log(topic_assignment)
    #log_separator()
    #log("Topic Counts:")
    #log(topic_counts)
    #log_separator()

    return topic_assignment, topic_counts, doc_counts, topic_N
    # RETURNS: topic_assignment, topic_counts, doc_counts, topic_N