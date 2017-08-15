#####################################
## Fix: Windows Warning Supression ##
#####################################

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#############
## Imports ##
#############

from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

######################
## Global Variables ##
######################

tokenizer = RegexpTokenizer(r'\w+')								# Regex for matching words made up of alphanumeric and underscore characters
stopWordsList = get_stop_words('en')							# English stop words list
stemmer = PorterStemmer()										# Tool for stemming tokens

##########
## Main ##
##########

def main():

	########################
	## Get Relevance Data ##
	########################

	# heading
	print("\n[Relevance Data]")
	# "Load Past Data"
	LDA_documents = ["I like to pet cats.",
						"cats are great pets.",
						"computers are my most favourite thing.",
						"a good program runs well on all computers.",
						"cats dont like dogs very much.",
						"cats could fly if they wanted to",
						"I left my computer running a program last night",
						"Programs really make my computer great"]
	# output
	for i in range(0, len(LDA_documents)):
		print(" - " + str(i) + ": " + LDA_documents[i])

	####################
	## Make LDA Model ##
	####################

	# make LDA model
	LDA_model = makeLDAModel(LDA_documents)

	#############################
	## Output Resulting Topics ##
	#############################

	# heading
	print("\n[Resulting Topics]")
	# print results
	for i in LDA_model.print_topics(num_topics=2, num_words=3):
		# report
		print(" - " + str(i[0]) + ": " + str(i[1]))

	#############
	## Testing ##
	#############

	# heading
	print("\n[Topic Distribution Test]")
	# load test document
	LDA_testDoc = ["cats are the best pet"]
	print(" - Test Document - '" + LDA_testDoc[0] + "'")
	# convert to required format
	LDA_testTokens = docsToTokens(LDA_testDoc)
	LDA_testDictionary = corpora.Dictionary(LDA_testTokens)
	LDA_testCorpus = [LDA_testDictionary.doc2bow(token) for token in LDA_testTokens]
	# determine topic distribution
	print(" - Topic Distribution:")
	for i in LDA_model.get_document_topics(LDA_testCorpus)[0]:
		print("    - Topic " + str(i[0]) + ": " + str(i[1]))

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

def makeLDAModel(LDA_documents):

	#############################
	## Generate LDA Primatives ##
	#############################

	# generate LDA tokens from documents
	LDA_tokens = docsToTokens(LDA_documents)
	# tokens to id-term dictionary
	dictionary = corpora.Dictionary(LDA_tokens)
	# tokens to document-term matrix
	corpus = [dictionary.doc2bow(token) for token in LDA_tokens]

	########################
	## Generate LDA Model ##
	########################

	# generate LDA model
	LDA_model = models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=50)

	##########
	## Done ##
	##########

	return LDA_model

#########################
## Program Entry Point ##
#########################

if (__name__=="__main__"):
	main()
