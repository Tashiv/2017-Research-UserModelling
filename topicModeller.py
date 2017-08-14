#########################
## Warning Supressions ##
#########################

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#############
## Imports ##
#############

import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

####################
## Configurations ##
####################

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')							# English stop words list

###############
## Main Loop ##
###############

def main():
	# make example data set
	number_of_topics = 2
	doc_set = ["I like to pet cats.",
				"cats are great pets.",
				"computers are my most favourite thing.",
				"a good program runs well on all computers.",
				"cats dont like dogs very much.",
				"cats could fly if they wanted to",
				"I left my computer running a program last night",
				"Programs really make my computer great"]

	# heading
	print("\n[Queries]")

	# output
	for i in range(0, len(doc_set)):
		print(" - " + str(i) + ": " + doc_set[i])

	# make LDA model
	LDA_Model = makeLDAModel(doc_set, number_of_topics)

	# heading
	print("\n[Resulting Topics]")

	# print results
	for i in LDA_Model:
		# report
		print(" - " + str(i[0]) + ": " + str(i[1]))

#############
## Methods ##
#############

def makeLDAModel(doc_set, number_of_topics):
	# initialize
	p_stemmer = PorterStemmer()
	texts = []

	# loop through document list
	for i in doc_set:
		# clean and tokenize document string
		raw = i.lower()
		tokens = tokenizer.tokenize(raw)
		# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in en_stop]
		# stem tokens
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		# add tokens to list
		texts.append(stemmed_tokens)

	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)

	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]

	# generate LDA model
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

	# check results
	return ldamodel.print_topics(num_topics=3, num_words=3)

#########################
## Program Entry Point ##
#########################

if (__name__=="__main__"):
	main()
