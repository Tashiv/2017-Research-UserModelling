"""
Caluculates symmetric Kullback-Leibler divergence.
@author Shingo OKAWA
@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
"""

#####################################
## Fix: Windows Warning Supression ##
#####################################

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#############
## Imports ##
#############

import numpy
import sys
from scipy import stats
from matplotlib import pyplot
from gensim import corpora, models, similarities, matutils
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

######################
## Global Variables ##
######################

tokenizer = RegexpTokenizer(r'\w+')								# Regex for matching words made up of alphanumeric and underscore characters
stopWordsList = get_stop_words('en')							# English stop words list
stemmer = PorterStemmer()										# Tool for stemming tokens

# Set training parameters.
passes = 50
iterations = 100
eval_every = None  # Don't evaluate model perplexity, takes too much time.
alpha = 0.01
eta = 0.01

##########
## Main ##
##########

def main():
    LDA_documents = ["desk desk desk",
                        "cat cat cat",
                        "computer computer computer",
                        "desk desk desk",
                        "cat cat cat",
                        "computer computer computer"]

    # initialize
    LDA_tokens = []
    # loop through document list
    for i in LDA_documents:
        # clean and tokenize document string
        tokens_cleaned = tokenizer.tokenize(i.lower())
        # remove stop words from tokens
        tokens_stopped = [i for i in tokens_cleaned if not i in stopWordsList]
        # stem tokens
        tokens_stemmed = [stemmer.stem(i) for i in tokens_stopped]
        # add tokens to list
        LDA_tokens.append(tokens_stemmed)
    print("Tokens", LDA_tokens, '\n')

    # tokens to id-term dictionary
    dictionary = corpora.Dictionary(LDA_tokens)
    print("Dictionary", dictionary, len(dictionary), '\n')

    # Holds token ids which appears only once.
    unique_ids = [token_id for token_id, frequency in dictionary.iteritems() if frequency == 1]
    # Filters out tokens which appears only once.
    dictionary.filter_tokens(unique_ids)
    # Filters out tokens which appears in more than no_above documents,
    # and keeps only the first keep_n tokens.
    ##dictionary.filter_extremes(no_above=5, keep_n=100000)
    # Compactifies.
    dictionary.compactify()
    print("Cleaned Dictionary", dictionary, len(dictionary), '\n')

    # Instanciates corpus.
    corpus = [dictionary.doc2bow(token) for token in LDA_tokens]
    print("corpus", corpus, '\n')


    # Caluculates symmetric KL divergence.
    kl_divergence = arun_metric(corpus, dictionary, max_topics=40)
    print('\n', kl_divergence)
    iOptimalTopics = find_Dip(kl_divergence)
    print("Optimal number of topics:", iOptimalTopics, '\n')
    # Plots KL divergence against number of topics.
    pyplot.plot(kl_divergence)
    pyplot.ylabel('Symmetric KL Divergence')
    pyplot.xlabel('Number of Topics')
    pyplot.savefig('kl_topics.png', bbox_inches='tight')

    # heading
    print("\n[Resulting Topics]")
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, \
                   alpha=alpha, eta=eta, \
                   iterations=iterations, num_topics=iOptimalTopics, \
                   passes=passes, eval_every=eval_every)
    # print results
    for i in lda.print_topics():
        # report
        print(" - " + str(i[0]) + ": " + str(i[1]))

    # heading
    print("\n[Actual Topics]")
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, \
                   alpha=alpha, eta=eta, \
                   iterations=iterations, num_topics=3, \
                   passes=passes, eval_every=eval_every)
    # print results
    for i in lda.print_topics():
        # report
        print(" - " + str(i[0]) + ": " + str(i[1]))

    bow_water = ['comput','comput','comput']
    bow = lda.id2word.doc2bow(bow_water) # convert to bag of words format first
    doc_topics, word_topics, phi_values = lda.get_document_topics(bow, per_word_topics=True)
    print("\comput:", doc_topics)

    bow_water = ['desk','desk','desk']
    bow = lda.id2word.doc2bow(bow_water) # convert to bag of words format first
    doc_topics, word_topics, phi_values = lda.get_document_topics(bow, per_word_topics=True)
    print("\desk:", doc_topics)

    bow_water = ['cat','cat','cat']
    bow = lda.id2word.doc2bow(bow_water) # convert to bag of words format first
    doc_topics, word_topics, phi_values = lda.get_document_topics(bow, per_word_topics=True)
    print("\cat:", doc_topics)

#############
## Methods ##
#############

def docsToTokens(LDA_documents):
	# initialize
	LDA_tokens = []
	# loop through document list
	for i in LDA_documents:
		# clean and tokenize document string
		tokens_cleaned = tokenizer.tokenize(i.lower())
		# remove stop words from tokens
		tokens_stopped = [i for i in tokens_cleaned if not i in stopWordsList]
		# stem tokens
		tokens_stemmed = [stemmer.stem(i) for i in tokens_stopped]
		# add tokens to list
		LDA_tokens.append(tokens_stemmed)
	# Done
	return LDA_tokens

##
# Finds the dip in the divergence values which indicates optimal number of topics
##
def find_Dip(kl_divergence):
    # sanity check for degenerate cases
    if len(kl_divergence) == 0:
        return 0
    elif len(kl_divergence) == 1:
        return 1
    elif len(kl_divergence) == 2:
        if kl_divergence[0] < kl_divergence[1]:
            return 0
        else:
            return 1
    # initialize
    iMin = 1000000
    iMinID = 0
    # check
    for i in range(1, len(kl_divergence), 1):
        # end case
        if i == len(kl_divergence)-1:
            if (kl_divergence[i] > kl_divergence[i-1]):
                continue
            # check to see if global minimum
            if kl_divergence[i] > iMin:
                continue
            # potential minimum
            iMin = kl_divergence[i]
            iMinID = i
        else:
            # check to see if local minimum
            if (kl_divergence[i] > kl_divergence[i+1]) or (kl_divergence[i] > kl_divergence[i-1]):
                continue
            # check to see if global minimum
            if kl_divergence[i] > iMin:
                continue
            # potential minimum
            iMin = kl_divergence[i]
            iMinID = i
    # Done
    return iMinID + 1

##
# Caluculates symmetric Kullback-Leibler divergence.
##
def symmetric_kl_divergence(p, q):
    print(len(p),"vs",len(q))
    return numpy.sum([stats.entropy(p, q), stats.entropy(q, p)])

##
# Caluculates Arun et al metric..
##
def arun_metric(corpus, dictionary, min_topics=1, max_topics=1, iteration=1):
    # initialize
    result = [];
    # Generates corpus length vectors.
    corpus_length_vector = numpy.array([sum(frequency for _, frequency in document) for document in corpus])
    # sanity check
    number_of_tokens = len(dictionary)
    if max_topics > number_of_tokens:
        max_topics = number_of_tokens
        print("Warning: Max_topics is more than number of tokens")
    # calculate KL divergence
    for i in range(min_topics, max_topics+1, iteration):
        # Instanciates LDA.
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, \
                       alpha=alpha, eta=eta, \
                       iterations=iterations, num_topics=i, \
                       passes=passes, eval_every=eval_every)
        # Caluculates raw LDA matrix.
        matrix = lda.expElogbeta
        # Caluculates SVD for LDA matris.
        U, document_word_vector, V = numpy.linalg.svd(matrix)
        # Gets LDA topics.
        lda_topics = lda[corpus]
        # Caluculates document-topic matrix.
        term_document_matrix = matutils.corpus2dense(
            lda_topics, lda.num_topics
        ).transpose()
        document_topic_vector = corpus_length_vector.dot(term_document_matrix)
        document_topic_vector = document_topic_vector + 0.0001
        document_topic_norm   = numpy.linalg.norm(corpus_length_vector)
        document_topic_vector = document_topic_vector / document_topic_norm
        result.append(symmetric_kl_divergence(document_word_vector, document_topic_vector))
    return result

#########################
## Program Entry Point ##
#########################

if __name__ == '__main__':
    main()
