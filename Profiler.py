######################################
## Fix: Windows Warning Suppression ##
######################################

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category = RuntimeWarning)

#############
## Imports ##
#############

import numpy
import math
import datetime
import multiprocessing
import ctypes
from scipy import stats
from matplotlib import pyplot
from gensim import corpora, models, matutils
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

######################
## Global Variables ##
######################

## LDA Parameters
LDA_SlidingWindow = 7																									# How old the log is allowed to be in days
LDA_Passes = 1
LDA_Iterations = 50
LDA_EvalEvery = None																									# Don't evaluate model perplexity, takes too much time.
LDA_Alpha = 'symmetric'
LDA_Eta = None
LDA_Minimum_Probability = 0.01
LDA_MaxTopics = 32
LDA_TopicRuns = 1																										# Number of times each topic KL_Divergence is evaluated, reduces noise
LDA_MaxThreads = 8
LDA_MinimumTimeFactorDifference = 0.01																					# Used in combining similar time factors
LDA_MinKLDivergenceDifference = 0.05

## Tools
fTokenizer = RegexpTokenizer(r'\w+')																					# Regex for matching words made up of alphanumeric and underscore characters
fStemmer = PorterStemmer()																								# Tool for stemming tokens
fStopWordsList = get_stop_words('en')																					# English stop words list

##########
## Main ##
##########

def main():
	# header
	print("[Topic Modeller]")
	# testing
	#runUnitTests()
	runOptimalTopicNumberTest()
	# done
	print("\n[DONE]")

#########################
## Evaluation Routines ##
#########################

def runOptimalTopicNumberTest():

	###############
	## load data ##
	###############

	logFile = "data/testlog-t10-d.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017,2,1,0,0,0))
	logItemCnt = len(logData)
	print("	- items:", logItemCnt)

	#########################
	## Generate Primitives ##
	#########################

	# initialize
	print("\n - Preparing Documents...")
	currentItem = 0
	documents = []
	# extract documents
	for i in logData:
		# load item
		documents.append(i[1])
		# report
		currentItem += 1
		if currentItem % 250 == 0:
			print("	- Processed " + str(currentItem) + " of " + str(logItemCnt))
	# Generate Primitives
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens)
	corpus = generateCorpus(dictionary, tokens)

	##################
	## Test Measure ##
	##################

	# multiple runs
	for x in range(3):

		# KL measures
		print("\n - Testing KL topic ranges:")
		measuresKL = calculateKLDivergences(corpus, dictionary, LDA_MaxTopics)
		optimalKLTopicsNN = findOptimalKLTopicCountNN(measuresKL)
		optimalKLTopicsGen = findOptimalKLTopicCountGen(measuresKL)
		print("    = NN Optimal Topic Count:", optimalKLTopicsNN)
		print("    = Gen Optimal Topic Count:", optimalKLTopicsGen)
		saveGraphY(measuresKL, "Number of Topics", "Measure",
				   "KL Div (ONN=" + str(optimalKLTopicsNN) + ", OGen=" + str(optimalKLTopicsGen) + ")",
				   "Graph_" + str(x) + "_KL")

		# JS Measures
		print("\n - Testing JS topic ranges:")
		measuresJS = calculateJSDivergences(corpus, dictionary, LDA_MaxTopics)
		optimalJSTopics = findOptimalJSTopicCount(measuresJS)
		print("    = Gen Optimal Topic Count:", optimalKLTopicsGen)
		saveGraphY(measuresKL, "Number of Topics", "Measure",
			   		"JS Div (O=" + str(optimalJSTopics) + ")",
			   		"Graph_" + str(x) + "_JS")

def runUnitTests():
	# test measures
	testKLMeasure()
	testSymmetricKLMeasure()
	testJSMeasure()

################
## Unit Tests ##
################

def testKLMeasure():
	"""
	@info Unit test to check accuracy of KL Divergence Measure.
	"""
	# report
	print("\n - Testing KL Measure (Should Increase)")
	# initialize
	setA = [0.1, 0.2, 0.3, 0.4]
	setB = [0.1, 0.2, 0.4, 0.3]
	setC = [0.4, 0.3, 0.2, 0.1]
	setD = [0.3, 0.3, 0.3, 0.1]
	setE = [0.4, 0.3, 0.2, 0.1]
	# test cases
	print("    - Accuracy Tests:")
	print("       -", setA, "vs", setA, "->", kl_divergence(setA, setA))
	print("       -", setA, "vs", setB, "->", kl_divergence(setA, setB))
	print("       -", setA, "vs", setC, "->", kl_divergence(setA, setC))
	print("    - Symmetry Tests:")
	print("       -", setD, "vs", setE, "->", kl_divergence(setD, setE))
	print("       -", setE, "vs", setD, "->", kl_divergence(setE, setD))

def testSymmetricKLMeasure():
	"""
	@info Unit test to check accuracy of Symmetric KL Divergence Measure.
	"""
	# report
	print("\n - Testing Symmetric KL Measure (Should Increase)")
	# initialize
	setA = [0.1, 0.2, 0.3, 0.4]
	setB = [0.1, 0.2, 0.4, 0.3]
	setC = [0.4, 0.3, 0.2, 0.1]
	setD = [0.3, 0.3, 0.3, 0.1]
	setE = [0.4, 0.3, 0.2, 0.1]
	# test cases
	print("    - Accuracy Tests:")
	print("       -", setA, "vs", setA, "->", symmetric_kl_divergence(setA, setA))
	print("       -", setA, "vs", setB, "->", symmetric_kl_divergence(setA, setB))
	print("       -", setA, "vs", setC, "->", symmetric_kl_divergence(setA, setC))
	print("    - Symmetry Tests:")
	print("       -", setD, "vs", setE, "->", symmetric_kl_divergence(setD, setE))
	print("       -", setE, "vs", setD, "->", symmetric_kl_divergence(setE, setD))

def testJSMeasure():
	"""
	@info Unit test to check accuracy of JS Divergence Measure.
	"""
	# report
	print("\n - Testing JS Measure (Should Increase)")
	# initialize
	setA = [0.1, 0.2, 0.3, 0.4]
	setB = [0.1, 0.2, 0.4, 0.3]
	setC = [0.4, 0.3, 0.2, 0.1]
	setD = [0.3, 0.3, 0.3, 0.1]
	setE = [0.4, 0.3, 0.2, 0.1]
	# test cases
	print("    - Accuracy Tests:")
	print("       -", setA, "vs", setA, "->", js_divergence(setA, setA))
	print("       -", setA, "vs", setB, "->", js_divergence(setA, setB))
	print("       -", setA, "vs", setC, "->", js_divergence(setA, setC))
	print("    - Symmetry Tests:")
	print("       -", setD, "vs", setE, "->", js_divergence(setD, setE))
	print("       -", setE, "vs", setD, "->", js_divergence(setE, setD))

##################################
## LDA Primitive Helper Methods ##
##################################

def generateTokensFromString(string):
	"""
	@info Converts a string into a set of cleaned/stemmed tokens.
	"""
	# convert to lower case and tokenize document string
	tokens_cleaned = fTokenizer.tokenize(string.lower())
	# FILTER 1: Stopwords
	tokens_stopped = [i for i in tokens_cleaned if not i in fStopWordsList]
	# FILTER 2: Stemming
	tokens_stemmed = [fStemmer.stem(i) for i in tokens_stopped]
	# done
	return tokens_stemmed

def generateTokens(documents):
	"""
	@info Converts an array of documents to a set of cleaned/stemmed tokens.
	"""
	# initialize
	tokens_final = []
	# generate set of tokens
	for i in documents:
		tokens_final.append(generateTokensFromString(i))
	# Done
	return tokens_final

def generateDictionary(tokens):
	"""
	@info Generates a dictionary from a set of tokens and cleans the dictionary if required.
	"""
	# generate dictionary
	dictionary = corpora.Dictionary(tokens)
	# FILTER 1: filter out uncommon tokens (appear only once)
	unique_ids = [token_id for token_id, frequency in dictionary.iteritems() if frequency == 1]
	dictionary.filter_tokens(unique_ids)
	## FILTER 2: filter out common tokens (appear in more than 5 documents)
	## dictionary.filter_extremes(no_above=5, keep_n=100000)
	# REPAIR: Reassign ids to 'fill gaps'
	dictionary.compactify()
	# done
	return dictionary

def generateCorpus(dictionary, tokens):
	"""
	@info Generates a corpus from a dictionary and a set of tokens.
	"""
	# generate corpus
	corpus = [dictionary.doc2bow(token) for token in tokens]
	# done
	return corpus

def generateLDAModel(corpus, dictionary, topicCount):
	"""
	@info Generates an LDA model from primitives.
	"""
	# build model
	return models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topicCount,
										iterations=LDA_Iterations, passes=LDA_Passes, eval_every=LDA_EvalEvery,
										alpha=LDA_Alpha, eta=LDA_Eta, minimum_probability=LDA_Minimum_Probability)

###########################
## LDA Reporting Methods ##
###########################

def printDocuments(documents):
	"""
	@info Prints raw document strings in a formatted way.
	"""
	# heading
	print("\n - Raw Documents:")
	# output
	for i in range(0, len(documents)):
		print("	- [DOC_" + str(i) + "] : " + documents[i])

def printTokens(tokens):
	"""
	@info Prints tokens in a formatted way.
	"""
	# heading
	print("\n - Generated Document Tokens:")
	# output
	for i in range(0, len(tokens)):
		print("	- [DOC_" + str(i) + "] : " + str(tokens[i]))

def printDictionary(dictionary):
	"""
	@info Prints an LDA dictionary in a formatted way.
	"""
	# heading
	print("\n - Generated Dictionary:")
	# output
	for i in dictionary:
		print("	- [DICT_" + str(i) + "] : " + dictionary[i])

def printCorpus(corpus):
	"""
	@info Prints an LDA corpus in a formatted way.
	"""
	# heading
	print("\n - Generated Corpus:")
	# output
	for i in range(0, len(corpus)):
		print("	- [DOC_" + str(i) + "] : " + str(corpus[i]))

def printMeasures(measures):
	"""
	@info Prints measure results in a formatted way.
	"""
	# heading
	print("\n - Topic Measure Results:")
	# output
	for i in range(0, len(measures)):
		print("	- [" + str(i+1) + "_TOPIC(S)] : " + str(measures[i]))

#################################
## LDA Usage Interface Methods ##
#################################

def classifyDocument(LDA_Model, document):
	"""
	@info Classifies a document using an LDA model and returns the matching topic.
	"""
	# convert to token format
	tokens = generateTokensFromString(document)
	bagOfWords = LDA_Model.id2word.doc2bow(tokens)
	# determine topics
	return LDA_Model.get_document_topics(bagOfWords, minimum_probability=0)

def extractTopicProbabilityDistributionFromModel(lda_model, dictionary, number_of_topics):
	"""
	@info Formats the topic distribution returned by Gensim.
	"""
	# initialize
	topics = []
	# extract topics
	for i in range(0, number_of_topics):
		# get topic distribution
		items = []
		for item in lda_model.show_topics(num_topics=-1, num_words=len(dictionary), formatted=False)[i][1]:
			items.append(item)
		# sort distribution
		items.sort(key=lambda tup: tup[0])
		# format into list of just probabilities
		topic = []
		for item in items:
			topic.append(item[1])
		topics.append(topic)
	# done
	return topics

###########################
## KL Divergence Methods ##
###########################

def symmetric_kl_divergence(p, q):
	"""
	@info Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	return numpy.sum([stats.entropy(p, q), stats.entropy(q, p)])

def calculateKLDivergence(corpus, dictionary, number_of_topics, result):
	"""
	@info   Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA, Modified by Tashiv Sewpersad
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	# report
	print("	   - Trying a topic count of " + str(number_of_topics) + "...")
	# initialize
	tempValues = []
	# run multiple times for better accuracy
	for j in range(0, LDA_TopicRuns):
		# Generates corpus length vectors.
		corpus_length_vector = numpy.array([sum(frequency for dictID, frequency in document) for document in corpus])
		# Instantiates LDA.
		lda_model = generateLDAModel(corpus, dictionary, number_of_topics)
		# Calculates raw LDA matrix.
		matrix = lda_model.expElogbeta
		# Calculates SVD for LDA matrix.
		U, document_word_vector, V = numpy.linalg.svd(matrix)
		# Gets LDA topics.
		lda_topics = lda_model[corpus]
		# Calculates document-topic matrix.
		term_document_matrix = matutils.corpus2dense(lda_topics, lda_model.num_topics).transpose()
		document_topic_vector = corpus_length_vector.dot(term_document_matrix)
		document_topic_vector = document_topic_vector + 0.00000001
		document_topic_norm = numpy.linalg.norm(corpus_length_vector)
		document_topic_vector = document_topic_vector / document_topic_norm
		# calculate KL divergence
		tempValues.append(symmetric_kl_divergence(document_word_vector, document_topic_vector))
	# average result
	result[number_of_topics - 1] = sum(tempValues) / len(tempValues)

def calculateKLDivergences(corpus, dictionary, max_topics=1):
	"""
	@info Calculates and returns KL Divergence values for topics counts from 1 to max_topics
	"""
	# sanity check
	token_count = len(dictionary)
	if max_topics > token_count:
		# report
		print("	* Warning: Max_topics is more than number of tokens, using " + str(token_count) + " instead of " + str(max_topics) + " as max topics.")
		# update
		max_topics = token_count
	# use correct threading mode
	if LDA_MaxThreads == 1:
		return calculateKLDivergencesST(corpus, dictionary, max_topics)
	else:
		return calculateKLDivergencesMT(corpus, dictionary, max_topics)

def calculateKLDivergencesST(corpus, dictionary, max_topics=1):
	"""
	@info Single Thread wrapper for calculateKLDivergences.
	"""
	# initialize
	result =  [None] * max_topics
	# calculate KL divergence
	print("	- Trying various KL topic counts: (max=" + str(max_topics) + ", SINGLE Thread Mode)")
	for i in range(0, max_topics, 1):
		# calculate divergence
		calculateKLDivergence(corpus, dictionary, i+1, result)
	# done
	return result

def calculateKLDivergencesMT(corpus, dictionary, max_topics=1):
	"""
	@info Multi Thread wrapper for calculateKLDivergences.
	"""
	# initialize
	result =  multiprocessing.Array(ctypes.c_double, max_topics)
	# calculate KL divergence
	print("	- Trying various KL topic counts: (max=" + str(max_topics) + ", MULTI-Threads=" + str(LDA_MaxThreads) + ")")
	jobs = []
	for i in range(0, max_topics, 1):
		# Create new threads
		worker = multiprocessing.Process(target=calculateKLDivergence, args=(corpus, dictionary, i+1, result,))
		worker.start()
		jobs.append(worker)
		# batch processing
		if len(jobs) >= LDA_MaxThreads:
			for j in jobs:
				j.join()
			jobs = []
	# wait for remaining threads to finish
	for j in jobs:
		j.join()
	# done
	return result

def findOptimalKLTopicCountNN(KL_Divergences):
	"""
	@info Finds the optimal topic number by looking for a local minimum based on a nearest-neighbour comparison.
	"""
	# sanity check for degenerate cases
	if len(KL_Divergences) < 2:
		return len(KL_Divergences)
	elif len(KL_Divergences) == 2:
		return 1
	# initialize
	iMin = float('inf')
	iMinID = 0
	# check
	for i in range(1, len(KL_Divergences), 1):
		# end case
		if i == len(KL_Divergences)-1:
			if KL_Divergences[i] > KL_Divergences[i-1] - LDA_MinKLDivergenceDifference:
				continue
			# check to see if global minimum
			if KL_Divergences[i] > iMin - LDA_MinKLDivergenceDifference:
				continue
			# potential minimum
			iMin = KL_Divergences[i]
			iMinID = i
		else:
			# check to see if local minimum
			if (KL_Divergences[i] > KL_Divergences[i+1] - LDA_MinKLDivergenceDifference
			or KL_Divergences[i] > KL_Divergences[i-1] - LDA_MinKLDivergenceDifference):
				continue
			# check to see if global minimum
			if KL_Divergences[i] > iMin - LDA_MinKLDivergenceDifference:
				continue
			# potential minimum
			iMin = KL_Divergences[i]
			iMinID = i
	# degenerate case
	if iMin == float('inf'):
		iMinID = 0
	# Done
	return iMinID + 1

def findOptimalKLTopicCountGen(KL_Divergences):
	"""
	@info Finds the optimal topic number by looking for the largest "dip" - i.e. considers more than immediate neighbours.
	"""
	# sanity check
	if len(KL_Divergences) == 0:
		return 0
	# initialize
	signs = []
	groupings = []
	scores = []
	# fill sign list
	for i in range(0, len(KL_Divergences)-1):
		if KL_Divergences[i] > KL_Divergences[i+1]:
			signs.append(1)
		elif KL_Divergences[i] < KL_Divergences[i+1]:
			signs.append(-1)
		else:
			signs.append(0)
	# fill groupings list
	currentSign = signs[0]
	groupings.append(1)
	for i in range(1, len(signs)):
		if signs[i] == currentSign or signs[i] == 0:
			groupings[len(groupings)-1] += 1
		else:
			groupings.append(1)
			currentSign = signs[i]
	# fill scores
	for i in range(0, len(groupings)-1):
		scores.append(min(groupings[i], groupings[i+1]))
	# find highest score
	bestScoreIndex = 0
	bestScore = scores[0]
	for i in range(1, len(scores)):
		if scores[i] > bestScore:
			bestScore = scores[i]
			bestScoreIndex = i
	# determine bottom index
	startIndex = 0
	for i in range(0, bestScoreIndex):
		startIndex += groupings[i]
	# determine top index
	endIndex = startIndex + groupings[bestScoreIndex] + groupings[bestScoreIndex+1] - 1
	# return average
	return round((startIndex + endIndex)/2)

#######################################
## Jensen-shannon divergence Measure ##
#######################################

def kl_divergence(p, q):
	"""
	@info Returns the Asymmetric KL Divergence of two sets.
	"""
	# sanity check
	if len(p) != len(q):
		return  float('inf')
	# distribution check
	if sum(p) < 0.99999999 or sum(p) > 1.00000001:
		print(" * Warning: KL Probability of P does not add up to 1.0. Instead it is:", sum(p))
	if sum(q) < 0.99999999 or sum(q) > 1.00000001:
		print(" * Warning: KL Probability of Q does not add up to 1.0. Instead it is:", sum(q))
	# calculate KL Divergence
	result = 0
	for i in range(0, len(p)):
		# special case check
		if q[i] == 0 or p[i] == 0:
			continue
		# add sum
		result += p[i] * math.log(p[i] / q[i], 2)
	# done
	return result

def js_divergence(p, q):
	"""
	@info Calculates the Jensen Shannon Divergence of two sets.
	"""
	# calculate M
	m = 0.5 * numpy.add(p, q)
	# calculate js divergence
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def calculatedJSDivergence(corpus, dictionary, number_of_topics, result):
	"""
	@info Calculates the JSD of a LDA model built from a given corpus and dictionary.
	"""
	# report
	print("	   - Trying a topic count of " + str(number_of_topics) + "...")
	# initialize
	tempValues = []
	# run multiple times for better accuracy
	for j in range(0, LDA_TopicRuns):
		# initialize
		lda_model = generateLDAModel(corpus, dictionary, number_of_topics)
		k_factor = 1
		if number_of_topics > 1:
			k_factor = 1 / (number_of_topics * (number_of_topics - 1))
		# build topics list
		topics = extractTopicProbabilityDistributionFromModel(lda_model, dictionary, number_of_topics)
		# compare resulting topics
		k_sum = 0
		for i in range(0, number_of_topics):
			for j in range(0, number_of_topics):
				if i >= j:
					continue
				m = topics[i]
				n = topics[j]
				k_sum += js_divergence(m, n)
		# multiply factor
		k_total = k_factor * k_sum
		# done
		tempValues.append(k_total)
	# average result
	result[number_of_topics - 1] = sum(tempValues) / len(tempValues)

def calculateJSDivergences(corpus, dictionary, max_topics=1):
	"""
	@info Calculates and returns JS  Divergence values for topics counts from 1 to max_topics
	"""
	# use correct threading mode
	if LDA_MaxThreads == 1:
		return calculateJSDivergencesST(corpus, dictionary, max_topics)
	else:
		return calculateJSDivergencesMT(corpus, dictionary, max_topics)

def calculateJSDivergencesST(corpus, dictionary, max_topics=1):
	"""
	@info Single Thread wrapper for calculatedJSDivergence.
	"""
	# initialize
	result =  [None] * max_topics
	# calculate KL divergence
	print("	- Trying various JS topic counts: (max=" + str(max_topics) + ", SINGLE Thread Mode)")
	for i in range(0, max_topics, 1):
		# calculate divergence
		calculatedJSDivergence(corpus, dictionary, i+1, result)
	# done
	return result

def calculateJSDivergencesMT(corpus, dictionary, max_topics=1):
	"""
	@info Multi Thread wrapper for calculatedJSDivergence.
	"""
	# initialize
	result =  multiprocessing.Array(ctypes.c_double, max_topics)
	# calculate KL divergence
	print("	- Trying various JS topic counts: (max=" + str(max_topics) + ", MULTI-Threads=" + str(LDA_MaxThreads) + ")")
	jobs = []
	for i in range(0, max_topics, 1):
		# Create new threads
		worker = multiprocessing.Process(target=calculatedJSDivergence, args=(corpus, dictionary, i+1, result,))
		worker.start()
		jobs.append(worker)
		# batch processing
		if len(jobs) >= LDA_MaxThreads:
			for j in jobs:
				j.join()
			jobs = []
	# wait for remaining threads to finish
	for j in jobs:
		j.join()
	# done
	return result

def findOptimalJSTopicCount(JS_Divergences):
	"""
	@info Finds optimal number of topics based on JS divergence.
	"""
	# find maximum
	maxDivID = 0
	maxDiv = JS_Divergences[0]
	for i in range(1, len(JS_Divergences)):
		if JS_Divergences[i] > maxDiv:
			maxDiv = JS_Divergences[i]
			maxDivID = i
	# done
	return maxDivID

#########################
## Time Factor Methods ##
#########################

def calculateTimeFactor(timestamp):
	"""
	@info Returns the timefactor representation for a given time.
	"""
	return abs((timestamp.hour*60+timestamp.minute)/1440)

def calculateWeighting(timeFactor1, timeFactor2):
	"""
	@info Used to compare to time  factors, close to one means more similar.
	"""
	return 1 - min(abs(timeFactor1 - timeFactor2), 1 - abs(timeFactor1 - timeFactor2))

def timeFactorToTimeString(timeFactor):
	"""
	@info Returns the string representation for a given timefactor.
	"""
	# extract parts
	hour = int((timeFactor * 1440) / 60)
	minute = int((timeFactor * 1440) % 60)
	# make string
	return str(hour) + ":" + str(minute)

def gatherTopicTimeFactors(LDA_Model, logData):
	"""
	@info Returns the raw timefactors for all topics in a model.
	"""
	# initialize
	timeFactors = dict()
	minimumProbability = 1 / len(LDA_Model.print_topics())
	# initialize list
	for i in range(0, len(LDA_Model.print_topics())):
		timeFactors[i] = [-1]
	# gather time factors
	for i in logData:
		# determine document's topic distribution
		topicDistributions = classifyDocument(LDA_Model, i[1])
		# process distribution info
		for topicDistribution in topicDistributions:
			# check if significant probability
			if topicDistribution[1] > minimumProbability:
				# store time factor
				if timeFactors[topicDistribution[0]] == [-1]:
					timeFactors[topicDistribution[0]].append(i[0])
				else:
					timeFactors[topicDistribution[0]] = [i[0]]
	# done
	return timeFactors

def cleanTimeFactors(timeFactors):
	"""
	@info Joins similar timefactors to simplify representation.
	"""
	# initialize
	cleanedTimeFactors = dict()
	# look at each topic's time factor
	for i in timeFactors:
		# remove irrelevant topics
		if timeFactors[i] == [-1]:
			continue
		# process timeFactors
		for timeFactor in timeFactors[i]:
			# initial insert
			if not(i in cleanedTimeFactors):
				cleanedTimeFactors[i] = [timeFactor]
				continue
			# check if similar element exists
			foundSimilarElement = False
			for k in range(0, len(cleanedTimeFactors[i])):
				if abs(cleanedTimeFactors[i][k] - timeFactor) < LDA_MinimumTimeFactorDifference:
					# average similarities
					cleanedTimeFactors[i][k] = (cleanedTimeFactors[i][k] + timeFactor) / 2.0
					# log finding
					foundSimilarElement = True
					break
			# add to list if no similarities
			if not foundSimilarElement:
				cleanedTimeFactors[i].append(timeFactor)
	# done
	return cleanedTimeFactors

def findMostRelevantTopic(currentTimeFactor, modelTimeFactors):
	"""
	@info Returns the most relevant topic based on a given timefactor.
	"""
	# initialize
	bestTopicID = -1
	bestWeighting = -1
	# find relevant topic
	for topicID in modelTimeFactors:
		for timeFactor in modelTimeFactors[topicID]:
			weighting = calculateWeighting(currentTimeFactor, timeFactor)
			if weighting > bestWeighting:
				bestTopicID = topicID
				bestWeighting = weighting
	# done
	return bestTopicID

##############
## File i/o ##
##############

def saveGraphXY(xValues, yValues, xAxisName, yAxisName, graphTitle, filename):
	"""
	@info Saves a graph of values to disk.
	"""
	# setup graph
	pyplot.close('all')
	pyplot.title(graphTitle)
	pyplot.plot(xValues, yValues, color="red")
	pyplot.ylabel(yAxisName, color="black")
	pyplot.xlabel(xAxisName, color="black")
	# save graph
	pyplot.savefig('data/' + filename + '.png', facecolor='white', edgecolor='white', bbox_inches='tight')

def saveGraphY(yValues, xAxisName, yAxisName, graphTitle, filename):
	"""
	@info Saves a graph of values to disk. This is a wrapper for GraphXY.
	"""
	# generate x axis values
	xValues = []
	for i in range(1, len(yValues) + 1):
		xValues.append(i)
	# plot graph
	saveGraphXY(xValues, yValues, xAxisName, yAxisName, graphTitle, filename)

def loadLogData(filename, referenceDate):
	"""
	@info Loads a query log into memory.
	"""
	# initialize
	result = []
	# read from file
	file = open(filename, "r")
	for line in file:
		# clean line
		line = line.strip()
		line = line.split("_#_")
		# process time part
		dateData = line[0].split(',')
		timestamp = datetime.datetime(int(dateData[0]), int(dateData[1]), int(dateData[2]), int(dateData[3]), int(dateData[4]), int(dateData[5]))
		# determine age of item
		if (referenceDate - timestamp).days >= LDA_SlidingWindow:
			continue
		# determine time factor
		timeFactor = calculateTimeFactor(timestamp)
		# generate record
		logItem = [timeFactor, line[1]]
		result.append(logItem)
	# close file
	file.close()
	# done
	return result

def saveUserProfile(LDA_Model, timeFactors, fileName):
	"""
	@info Generates and Saves a user profile to disk.
	"""
	# initialize
	file = open(fileName, "w")
	# write topics
	for topic in LDA_Model.print_topics():
		# write time Factors
		if topic[0] in timeFactors:
			file.write(str(timeFactors[topic[0]]))
			# write topic keywords
			file.write(" [")
			topicKeywords = topic[1].split(" + ")
			for i in range(0, len(topicKeywords)):
				# write keyword
				topicKeyword = topicKeywords[i].split("*")
				file.write("(" + topicKeyword[0] + ", " + topicKeyword[1] + ")")
				# space with comma
				if i < len(topicKeywords)-1:
					file.write(", ")
			file.write("]\n")
	# close file
	file.close()

#############################
## Chrome Interface Plugin ##
#############################

def buildProfile():
	"""
	@info Used to build a user profile for the Chrome Plugin.
	"""
	###################
	## Load Log Data ##
	###################

	# load data
	print("\n - Loading Log Data...")
	logFile = "file-log.txt"
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime.now())
	print("	- items:", len(logData))

	# extract documents
	documents = []
	for i in logData:
		documents.append(i[1])

	# Generate Primitives
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens)
	corpus = generateCorpus(dictionary, tokens)

	#####################
	## Build LDA Model ##
	#####################

	# calculate divergences
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printMeasures(KL_Divergences)
	# final optimal topic count
	optimalTopicCnt = findOptimalKLTopicCountNN(KL_Divergences)
	print("\n - Optimal number of topics:", optimalTopicCnt)
	# generate lda model
	LDA_Model = generateLDAModel(corpus, dictionary, optimalTopicCnt)

	#############################
	## Topics Time Association ##
	#############################

	# generate time factor data
	timeFactors = gatherTopicTimeFactors(LDA_Model, logData)
	cleanedTimeFactors = cleanTimeFactors(timeFactors)

	# print time factors
	print("\n - Resulting Time Factors:")
	for i in cleanedTimeFactors:
		# print data
		print("	- TOPIC_" + str(i) + ": " + str(cleanedTimeFactors[i]), end="")
		# print string version
		print(" | [" , end = "")
		for j in range(0, len(cleanedTimeFactors[i])):
			print(timeFactorToTimeString(cleanedTimeFactors[i][j]), end="")
			if j < len(cleanedTimeFactors[i]) - 1:
				print(", ", end="")
		print("]")

	#####################
	## Save to profile ##
	#####################

	logFilename = "log-profile.txt"
	print("\n - saving user profile to " + logFile + "...")
	saveUserProfile(LDA_Model, cleanedTimeFactors, logFilename)

#########################
## Program Entry Point ##
#########################

if __name__ == '__main__':
	main()