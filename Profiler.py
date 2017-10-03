#### Tashiv Sewpersad
#### Completed 03 - 10 - 2017
#### A program for modelling users based on time and topic.

######################################
## Fix: Windows Warning Suppression ##
######################################

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category = RuntimeWarning)

#############
## Imports ##
#############

import sys
import numpy
import math
import datetime
import multiprocessing
import ctypes
from pathlib import Path
from Profiler_Grapher import StackedBarGrapher
from scipy import stats
from matplotlib import pyplot
from gensim import corpora, models, matutils
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

######################
## Global Variables ##
######################

## Topic Modelling Parameters
LDA_SlidingWindow = 7																									# How old the log is allowed to be in days
LDA_FallbackItemCnt = 50																								# Number of items to load if no recent items found
LDA_Passes = 100																										# correct: 100
LDA_Iterations = 250																									# correct: 250
LDA_EvalEvery = None																									# Don't evaluate model perplexity, takes too much time.
LDA_Alpha = "quad_symmetric"																							# correct: "quad_symmetric"
LDA_Eta = "quad_symmetric"																								# correct: "quad_symmetric"
LDA_Minimum_Probability = 0.01
LDA_MaxTopics = 16
LDA_TopicRuns = 8																										# Number of times each topic KL_Divergence is evaluated, reduces noise
LDA_MaxThreads = 8
LDA_MinKLDivergenceDifference = 0.05
LDA_EffectiveZero = 0.00000001

## Time Modelling Parameters
LDA_MinimumTimeFactorDifference = 0.01																					# Used in combining similar time factors
LDA_TimeScale = 0																										# 0 = day scale, 1 = 3 day scale, 2 = week scale

## Evaluation Parameters
EVAL_runs = 8

## Tools
fTokenizer = RegexpTokenizer(r'\w+')																					# Regex for matching words made up of alphanumeric and underscore characters
fStemmer = PorterStemmer()																								# Tool for stemming tokens
fStopWordsList = get_stop_words('en')																					# English stop words list

##########
## Main ##
##########

def main():
	"""
	@info Decides if the program should run in testing mode or user modelling mode.
	"""
	# header
	print("[Topic Modeller]")
	# check mode
	if len(sys.argv) == 2:																								# Benchmark Mode
		if sys.argv[1] == "-b":
			# initialize
			options = [' '] * 12
			# render loop
			while True:
				# print menu
				printTestingOptions(options)
				# interpret input
				response = input("\n > Enter action: ")
				if response.isdigit():
					if 1 <= int(response) <= 12:
						if options[int(response)-1] ==  ' ':
							options[int(response)-1] = 'X'
						else:
							options[int(response)-1] = ' '
					else:
						print(" X Invalid Option")
				elif response == "p":
					buildProfile()
				elif response == "g":
					runTestingOptions(options)
				elif response == "c":
					options = [' '] * 12
				elif response == "q":
					break
				else:
					print(" X Invalid Option")
		else:
			print("\n - Only supports a lone '-b' parameter.")
	else:																												# Plugin Mode
		buildProfile()
	# done
	print("\n[DONE]")

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
	# check for empty primitives
	if len(corpus) == 0 or len(dictionary) == 0:
		dictionary = { 0 : "NODATA"}
		corpus = [[(0,1)]]
	# custom parameters
	if LDA_Alpha == "quad_symmetric":
		temp_LDA_Alpha = 1 / math.pow(topicCount, 2)
	else:
		temp_LDA_Alpha = LDA_Alpha
	if LDA_Eta == "quad_symmetric":
		temp_LDA_Eta = 1 / math.pow(topicCount, 2)
	else:
		temp_LDA_Eta = LDA_Eta
	# build model
	return models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topicCount,
										iterations=LDA_Iterations, passes=LDA_Passes, eval_every=LDA_EvalEvery,
										alpha=temp_LDA_Alpha, eta=temp_LDA_Eta, minimum_probability=LDA_Minimum_Probability)

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

def printTopicDistributions(model, dictionary):
	"""
	@info Prints topic-word distribution sets in a formatted way.
	"""
	# heading
	print("\n - Topic Distribution")
	# output
	topicNumber = 1
	topicWords = extractTopicWordsFromModel(model, dictionary)
	for topic in extractTopicProbabilityDistributionFromModel(model, dictionary):
		# report topic number
		print("    - " + str(topicNumber), end=": | ")
		# format
		wordNumber = 0
		for topicItem in topic:
			print(topicWords[wordNumber] + "=" + format(topicItem * 100, ".1f").rjust(5) + "%", end=" | ")
			wordNumber += 1
		print()
		# track topic
		topicNumber += 1

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

def extractTopicProbabilityDistributionFromModel(lda_model, dictionary, mustFilter=False):
	"""
	@info Formats the topic distribution returned by Gensim.
	"""
	# initialize
	topics = []
	# extract topics
	for i in range(0, lda_model.num_topics):
		# get topic distribution
		items = []
		for item in lda_model.show_topics(num_topics=-1, num_words=len(dictionary), formatted=False)[i][1]:
			items.append(item)
		# sort distribution
		items.sort(key=lambda tup: tup[0])
		# format into list of just probabilities
		topic = []
		for item in items:
			if mustFilter and (item[1] <= (1 / lda_model.num_topics) + LDA_EffectiveZero):
				topic.append(LDA_EffectiveZero)
			else:
				topic.append(item[1])
		topics.append(topic)
	# done
	return topics

def extractTopicWordsFromModel(lda_model, dictionary):
	"""
	@info Formats the topic distribution returned by Gensim with word names.
	"""
	# extract topics
	items = []
	for item in lda_model.show_topics(num_topics=-1, num_words=len(dictionary), formatted=False)[0][1]:
		items.append(item)
	# sort distribution
	items.sort(key=lambda tup: tup[0])
	# format into list of just probabilities
	topicWords = []
	for item in items:
		topicWords.append(item[0])
	# done
	return topicWords

def getRelevantTopicKeyWords(lda_model, dictionary, topic_ID):
	"""
	@info A method for getting only the relevant words from a topic distribution.
	"""
	# determine number of words
	wordCount = max(len(dictionary), 1)
	# initialize
	topic = lda_model.print_topic(topic_ID, topn=100)
	relevantWords = []
	# extract words
	for topicWord in topic.split(" + "):
		# extract components
		topicWordComponents = topicWord.split("*")
		word_probability = float(topicWordComponents[0])
		word_Value = topicWordComponents[1][1:-1]
		# check if significant
		if word_probability >= round((1.0 / wordCount)-LDA_EffectiveZero, 3):
			relevantWords.append(word_Value)
	# done
	return relevantWords

###########################
## KL Divergence Methods ##
###########################

def symmetric_kl_divergence(p, q):
	"""
	@info Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA, Modified by Tashiv Sewpersad
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	if len(p) != len(q):
		return float("inf")
	else:
		return numpy.sum([stats.entropy(p, q), stats.entropy(q, p)])

def calculateKLDivergence(corpus, dictionary, number_of_topics, result):
	"""
	@info   Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA, Modified by Tashiv Sewpersad
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	# sanity check
	if len(dictionary) == 0 or len(corpus) == 0:
		return 0
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
		document_topic_vector = document_topic_vector + LDA_EffectiveZero
		document_topic_norm = numpy.linalg.norm(corpus_length_vector)
		document_topic_vector = document_topic_vector / document_topic_norm
		# padding
		padded_document_word_vector = []
		for x in range(len(document_word_vector)):
			padded_document_word_vector.append(document_word_vector[x])
		for x in range(len(document_topic_vector) - len(document_word_vector)):
			padded_document_word_vector.append(LDA_EffectiveZero)
		# calculate KL divergence
		tempValues.append(symmetric_kl_divergence(padded_document_word_vector, document_topic_vector))
	# average result
	result[number_of_topics - 1] = sum(tempValues) / len(tempValues)

def calculateKLDivergences(corpus, dictionary, max_topics=1):
	"""
	@info Calculates and returns KL Divergence values for topics counts from 1 to max_topics
	"""
	# use correct threading mode
	result = []
	if LDA_MaxThreads == 1:
		result = calculateKLDivergencesST(corpus, dictionary, max_topics)
	else:
		result = calculateKLDivergencesMT(corpus, dictionary, max_topics)
	# find largest value
	iLargestValue = 0
	for i in result:
		if not i == float('inf'):
			iLargestValue = max(i, iLargestValue)
	# bound values
	for i in range(len(result)):
		if result[i] == float('inf'):
			result[i] = iLargestValue * (i+1)
	# done
	return result

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
	if len(KL_Divergences) <= 1:
		return 1
	# initialize
	signs = []
	groupings = []
	scores = []
	# fill sign list
	currentSign = 1
	for i in range(0, len(KL_Divergences)-1):
		if KL_Divergences[i] > KL_Divergences[i+1]:
			signs.append(-1)
			currentSign = -1
		elif KL_Divergences[i] < KL_Divergences[i+1]:
			signs.append(1)
			currentSign = 1
		else:
			signs.append(currentSign)
	# filter sign list
	filtered_signs = []
	for i in range(0, len(signs)-1):
		# side offsets
		if i < 2 or i >= len(signs) - 2:
			filtered_signs.append(signs[i])
			continue
		# compare [...,_,_,X,X,O,X,X,_,_,...]
		if filtered_signs[i-2] == filtered_signs[i-1] and filtered_signs[i-1] == signs[i+1] and signs[i+1] == signs[i+2]:
			filtered_signs.append(filtered_signs[i-2])
		else:
			filtered_signs.append(signs[i])
	# fill groupings list
	currentSign = filtered_signs[0]
	groupings.append(currentSign)
	for i in range(1, len(filtered_signs)):
		if filtered_signs[i] == currentSign or filtered_signs[i] == 0:
			groupings[len(groupings)-1] += 1
		else:
			groupings.append(1)
			currentSign = filtered_signs[i]
	# special case
	if len(groupings) == 1:
		return 1
	# fill scores
	for i in range(0, len(groupings)-1):
		scores.append(min(groupings[i], groupings[i+1]))
	# find highest score
	bestScoreIndex = 0
	bestScore = scores[0]
	signType = filtered_signs[0]
	for i in range(1, len(scores)):
		# invert sign
		signType = signType * -1
		# check for local minimum
		if scores[i] > bestScore and signType == -1:
			bestScore = scores[i]
			bestScoreIndex = i
	# determine bottom index
	startIndex = 0
	for i in range(0, bestScoreIndex):
		startIndex += groupings[i]
	# determine top index
	inflectionIndex = startIndex + groupings[bestScoreIndex] + 1
	# return average
	return round(inflectionIndex)

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
	## distribution check
	#if sum(p) < 1-LDA_EffectiveZero or sum(p) > 1+LDA_EffectiveZero:
	#	print(" * Warning: KL Probability of P does not add up to 1.0. Instead it is:", sum(p))
	#if sum(q) < 1-LDA_EffectiveZero or sum(q) > 1+LDA_EffectiveZero:
	#	print(" * Warning: KL Probability of Q does not add up to 1.0. Instead it is:", sum(q))
	# calculate KL Divergence
	result = 0
	for i in range(0, len(p)):
		# special case check
		if q[i] == 0:
			q[i] = LDA_EffectiveZero
		if p[i] == 0:
			p[i] = LDA_EffectiveZero
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
	return (0.5 * kl_divergence(p, m)) + (0.5 * kl_divergence(q, m))

def calculatedJSDivergence(corpus, dictionary, number_of_topics, result):
	"""
	@info Calculates the JSD of a LDA model built from a given corpus and dictionary.
	"""
	# report
	print("	   - Trying a topic count of " + str(number_of_topics) + "...")
	# degenerate case
	if number_of_topics == 0 or number_of_topics == 1:
		return 0
	# initialize
	tempValues = []
	# run multiple times for better accuracy
	for x in range(0, LDA_TopicRuns):
		# initialize
		lda_model = generateLDAModel(corpus, dictionary, number_of_topics)
		k_factor = 1
		if number_of_topics > 1:
			k_factor = 1.0 / (number_of_topics * (number_of_topics-1))
		# build topics list
		topics = extractTopicProbabilityDistributionFromModel(lda_model, dictionary)
		# compare resulting topics
		k_sum = 0
		for m in range(0, number_of_topics):
			for n in range(0, number_of_topics):
				k_sum += js_divergence(topics[m], topics[n])
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
	return maxDivID + 1

#############################
## Cosine Distance Measure ##
#############################

def cosine_distance(i, j):
	"""
	@info Calculates the Cosine Distance of two sets.
	"""
	# sanity check
	if len(i) != len(j):
		return float("inf")
	# process sum part of measure
	numerator = 0
	sum_p_squared = LDA_EffectiveZero
	sum_q_squared = LDA_EffectiveZero
	for v in range(len(i)):
		# accumulate numerator component
		numerator += i[v] * j[v]
		# accumulate denominator components
		sum_p_squared += math.pow(i[v], 2)
		sum_q_squared += math.pow(j[v], 2)
	# combine components
	return numerator / (math.pow(sum_p_squared, 0.5) * math.pow(sum_q_squared, 0.5))

def calculateCosineDistance(corpus, dictionary, number_of_topics, result):
	"""
	@info Calculates the Cosine Distance of a LDA model built from a given corpus and dictionary.
	"""
	# report
	print("	   - Trying a topic count of " + str(number_of_topics) + "...")
	# degenerate case
	if number_of_topics == 0 or number_of_topics == 1:
		return 0
	# initialize
	tempValues = []
	# run multiple times for better accuracy
	for x in range(0, LDA_TopicRuns):
		# initialize
		lda_model = generateLDAModel(corpus, dictionary, number_of_topics)
		# build topics list
		topics = extractTopicProbabilityDistributionFromModel(lda_model, dictionary)
		# sort
		#for i in range(len(topics)):
		#	topics[i] = sorted(topics[i])
		# compare resulting topics
		distance_denominator = (number_of_topics * (number_of_topics - 1)) * 0.5
		distance_numerator = 0
		for i in range(0, number_of_topics):
			for j in range(i+1, number_of_topics):
				distance_numerator += cosine_distance(topics[i], topics[j])
		# determine distance
		distance_total = distance_numerator / distance_denominator
		# done
		tempValues.append(distance_total)
	# average result
	result[number_of_topics - 1] = sum(tempValues) / len(tempValues)

def calculateCosineDistances(corpus, dictionary, max_topics=1):
	"""
	@info Calculates and returns Cosine Distance values for topics counts from 1 to max_topics
	"""
	# use correct threading mode
	if LDA_MaxThreads == 1:
		return calculateCosineDistancesST(corpus, dictionary, max_topics)
	else:
		return calculateCosineDistancesMT(corpus, dictionary, max_topics)

def calculateCosineDistancesST(corpus, dictionary, max_topics=1):
	"""
	@info Single Thread wrapper for calculateCosineDistance.
	"""
	# initialize
	result =  [None] * max_topics
	# calculate KL divergence
	print("	- Trying various CD topic counts: (max=" + str(max_topics) + ", SINGLE Thread Mode)")
	for i in range(0, max_topics, 1):
		# calculate divergence
		calculateCosineDistance(corpus, dictionary, i+1, result)
	# done
	return result

def calculateCosineDistancesMT(corpus, dictionary, max_topics=1):
	"""
	@info Multi Thread wrapper for calculateCosineDistance.
	"""
	# initialize
	result =  multiprocessing.Array(ctypes.c_double, max_topics)
	# calculate KL divergence
	print("	- Trying various CD topic counts: (max=" + str(max_topics) + ", MULTI-Threads=" + str(LDA_MaxThreads) + ")")
	jobs = []
	for i in range(0, max_topics, 1):
		# Create new threads
		worker = multiprocessing.Process(target=calculateCosineDistance, args=(corpus, dictionary, i+1, result,))
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

def findOptimalCDTopicCount(CDistances):
	"""
	@info Finds optimal number of topics based on Cosine Distance.
	"""
	# degenerate cases
	if len(CDistances) == 1:
		return 1
	# find maximum
	minDivID = 1
	minDiv = CDistances[1]
	for i in range(2, len(CDistances)):
		if CDistances[i] < minDiv:
			minDiv = CDistances[i]
			minDivID = i
	# done
	return minDivID + 1

#########################
## Time Factor Methods ##
#########################

def getTimeScaleAsString():
	"""
	@info Gets the string representation of the current LDA modelling process's time scale.
	"""
	if LDA_TimeScale == 0:
		return "1DAY"
	elif LDA_TimeScale == 1:
		return "3DAY"
	elif LDA_TimeScale == 2:
		return "WEEK"
	else:
		return "UNKNOWN"

def calculateTimeFactor(timestamp):
	"""
	@info Returns the timefactor representation for a given time.
	"""
	if LDA_TimeScale == 0:
		return abs((timestamp.hour*60+timestamp.minute)/1440)
	elif LDA_TimeScale == 1:
		return abs(((timestamp.weekday()%3)*1440+ timestamp.hour*60+timestamp.minute)/4320)
	elif LDA_TimeScale == 2:
		return abs((timestamp.weekday()*1440+timestamp.hour*60+timestamp.minute)/10080)
	else:
		return -1

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
	@info Returns the raw time_factors for all topics in a model.
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
			if topicDistribution[1] > minimumProbability or topicDistribution[1] >= 1.0 - LDA_EffectiveZero:
				# store time factor
				if timeFactors[topicDistribution[0]] == [-1]:
					timeFactors[topicDistribution[0]] = [i[0]]
				else:
					timeFactors[topicDistribution[0]].append(i[0])
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

def saveGraphOfTopics(rawTopicDistribution, topicWords, graphTitle, filename):
	"""
	@info Based on the code from https://github.com/minillinim/stackedBarGraph
	"""
	# initialize
	SBG = StackedBarGrapher()
	# gather parameters
	d = numpy.array(rawTopicDistribution)
	d_widths = [1] * len(rawTopicDistribution)
	d_labels = [str(i+1) for i in range(len(rawTopicDistribution))]
	d_colors = [pyplot.cm.coolwarm(i / len(rawTopicDistribution[0]), 1) for i in range(len(rawTopicDistribution[0]))]
	# generate plot
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	plots = SBG.stackedBarPlot(ax, d, d_colors, xLabels=d_labels, yTicks=len(rawTopicDistribution), widths=d_widths, scale=False)
	# add plot titles
	pyplot.title(graphTitle)
	pyplot.ylabel("Probability")
	pyplot.xlabel("Topic")
	# prepare plot for legend
	fig.subplots_adjust(bottom=0.4)
	pyplot.tight_layout()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# add legend
	pyplot.legend([p[0] for p in reversed(plots)],
				  [i for i in reversed(topicWords)],
				  bbox_to_anchor=(1.0, 1.0),
				  fancybox=True,
				  shadow=True)
	# save
	pyplot.savefig('data/' + filename + '.png', facecolor='white', edgecolor='white', bbox_inches='tight')

def loadLogData(filename, referenceDate):
	"""
	@info Loads a query log into memory.
	"""
	# initialize
	fallbackResults = []
	result = []
	# read from file
	try:
		file = open(filename, "r")
		for line in file:
			# clean line
			line = line.strip()
			line = line.split("_#_")
			# process time part
			dateData = line[0].split(',')
			timestamp = datetime.datetime(int(dateData[0]), int(dateData[1]), int(dateData[2]), int(dateData[3]), int(dateData[4]), int(dateData[5]))
			timeFactor = calculateTimeFactor(timestamp)
			logItem = [timeFactor, line[1]]
			# determine age of item
			if (referenceDate - timestamp).days >= LDA_SlidingWindow:
				fallbackResults.append(logItem)
			else:
				result.append(logItem)
		# close file
		file.close()
		# check if any relevant items were found
		if len(result) == 0:
			counter = 0
			for i in range(len(fallbackResults)):
				# get an item
				result.append(fallbackResults[len(fallbackResults) - 1 - i])
				counter += 1
				# bound usage
				if counter == LDA_FallbackItemCnt:
					break
	except:
		print("\n *** Warning: Cannot access '" + filename + "' ***\n")
		result = []
	# done
	return result

def saveUserProfile(LDA_Model, dictionary, timeFactors, fileName):
	"""
	@info Generates and Saves a user profile to disk.
	"""
	# initialize
	file = open(fileName, "w")
	# write topics
	for topic in LDA_Model.print_topics():
		# write time Factors
		if topic[0] in timeFactors:
			file.write(str([round(i,3) for i in timeFactors[topic[0]]]))
			# write topic keywords
			file.write(" [")
			topicKeywords = topic[1].split(" + ")
			written = False
			for i in range(0, len(topicKeywords)):
				# write keyword
				topicKeyword = topicKeywords[i].split("*")
				if float(topicKeyword[0]) > (1 / len(dictionary)) - LDA_EffectiveZero:
					# space by comma
					if written:
						file.write(",")
					# write item
					file.write("(" + str(round(float(topicKeyword[0]),3)) + ", " + topicKeyword[1] + ")")
					written = True
			file.write("]\n")
	# close file
	file.close()

###################
## Core Routines ##
###################

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
	measures = calculateJSDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printMeasures(measures)
	# final optimal topic count
	optimalTopicCnt = findOptimalJSTopicCount(measures)
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

	# determine filename
	flag = 0
	flagFilename = "file-flag.txt"
	flagFilePath = Path(flagFilename)
	if flagFilePath.is_file():
		# get current flag
		flagFile = open(flagFilename, 'r')
		flag = int(flagFile.readline().strip())
		flagFile.close()
		# invert
		flag = (flag + 1) % 2
	# add to log
	print("\n - saving user profile to " + logFile + "...")
	saveUserProfile(LDA_Model, dictionary, cleanedTimeFactors, "file-profile" + str(flag) + ".txt")
	# show that profile is ready
	flagFile = open(flagFilename, 'w')
	flagFile.write(str(flag))
	flagFile.close()

def printTestingOptions(options):
	"""
	@info Prints the menu for the testing mode.
	"""
	# initialize
	prefix = "       - "
	# header
	print("\n - Testing Options:")
	# options
	print("    - Unit Testing:")
	print(prefix + " 1 : [" + options[0]  + "] runUnitTests                                     : Check how the measures differentiate different sets from one another.")
	print(prefix + " 2 : [" + options[1]  + "] benchmarkJSMeasure                               : Check JS measure's behaviour for a 5-topic distinct synthetic log.")
	print(prefix + " 3 : [" + options[2]  + "] benchmarkCDMeasure                               : Check CD measure's behaviour for a 5-topic distinct synthetic log.")
	print(prefix + " 4 : [" + options[3]  + "] benchmarkKLMeasure                               : Check KL measure's behaviour for a 5-topic distinct synthetic log.")
	print("    - General Testing:")
	print(prefix + " 5 : [" + options[4]  + "] runModelAccuracyTest                             : Examine the quality of models created with the correct topic count.")
	print(prefix + " 6 : [" + options[5]  + "] runOptimalTopicNumberTest                        : Examine the topic predictions for all measures for a 10-topic log.")
	print(prefix + " 7 : [" + options[6]  + "] runModellingTest                                 : Test document classification of the LDA modelling process.")
	print("    - Measure Evaluation:")
	print(prefix + " 8 : [" + options[7]  + "] evaluateMeasurePerformanceDistinct               : Evaluates the measure accuracies for a set of 5 distinct synthetic logs.")
	print(prefix + " 9 : [" + options[8]  + "] evaluateMeasurePerformanceRealistic              : Evaluates the measure accuracies for a set of 5 realistic synthetic logs")
	print("    - Profile Evaluation:")
	print(prefix + "10 : [" + options[9]  + "] evaluatePredictionCapabilityForDailyUser         : Tests prediction performance for a single time scale on a Daily User.")
	print(prefix + "11 : [" + options[10] + "] evaluatePredictionCapabilityForWeeklyUser        : Tests prediction performance for a single time scale on a Weekly User.")
	print(prefix + "12 : [" + options[11] + "] evaluatePredictionCapabilityForAllTimeScales     : Tests prediction performance for all time scales on a Daily and Weekly User.")
	print("    - Other:")
	print(prefix + " p : build profile")
	print(prefix + " g : run actions")
	print(prefix + " c : clear selections")
	print(prefix + " q : quit")

def runTestingOptions(options):
	"""
	@info Interprets the test mode options and runs the corresponding tests.
	"""

	####################
	## Initialization ##
	####################

	# initialize
	activatedCharacter = 'X'
	# header
	print("\n=============================== TESTING MODE START ===============================")

	##################
	## Unit Testing ##
	##################

	if options[0] == activatedCharacter:
		testKLMeasure()
		testSymmetricKLMeasure()
		testJSMeasure()
		testCDMeasure()
	if options[1] == activatedCharacter:
		benchmarkJSMeasure()
	if options[2] == activatedCharacter:
		benchmarkCDMeasure()
	if options[3] == activatedCharacter:
		benchmarkKLMeasure()

	#####################
	## General Testing ##
	#####################

	if options[4] == activatedCharacter:
		runModelAccuracyTest()
	if options[5] == activatedCharacter:
		runOptimalTopicNumberTest()
	if options[6] == activatedCharacter:
		runModellingTest()

	########################
	## Measure Evaluation ##
	########################

	if options[7] == activatedCharacter:
		evaluateMeasurePerformanceDistinct()
	if options[8] == activatedCharacter:
		evaluateMeasurePerformanceRealistic()

	########################
	## Profile Evaluation ##
	########################

	if options[9] == activatedCharacter:
		evaluatePredictionCapabilityForDailyUser()
	if options[10] == activatedCharacter:
		evaluatePredictionCapabilityForWeeklyUser()
	if options[11] == activatedCharacter:
		evaluatePredictionCapabilityForAllTimeScales()

	# header
	print("\n=============================== TESTING MODE END =================================")

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
	setI = [0.4, 0.3, 0.2, 0.1]
	# test cases
	print("    - Accuracy Tests:")
	print("       -", setA, "vs", setA, "->", kl_divergence(setA, setA))
	print("       -", setA, "vs", setB, "->", kl_divergence(setA, setB))
	print("       -", setA, "vs", setC, "->", kl_divergence(setA, setC))
	print("    - Symmetry Tests:")
	print("       -", setD, "vs", setI, "->", kl_divergence(setD, setI))
	print("       -", setI, "vs", setD, "->", kl_divergence(setI, setD))
	# behaviour tests
	setF = [0.25, 0.25, 0.25, 0.25]
	setG = [0.50, 0.50, 0.00, 0.00]
	setH = [0.00, 0.00, 0.50, 0.50]
	print("    - Range Tests:")
	print("       -", setG, "vs", setG, "->", cosine_distance(setG, setG))
	print("       -", setF, "vs", setF, "->", cosine_distance(setF, setF))
	print("       -", setF, "vs", setG, "->", cosine_distance(setF, setG))
	print("       -", setH, "vs", setG, "->", cosine_distance(setH, setG))
	print("       -", setH, "vs", setF, "->", cosine_distance(setH, setF))
	# relative divergence
	print("    - Comparative Tests:")
	setI = [0.25, 0.25, 0.25, 0.25]
	setJ = [0.24, 0.26, 0.24, 0.26]
	setK = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
	setL = [0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09]
	print("       -", setI, "vs", setJ, "->", kl_divergence(setI, setJ))
	print("       -", setK, "vs", setL, "->", kl_divergence(setK, setL))


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
	# behaviour tests
	setF = [0.25, 0.25, 0.25, 0.25]
	setG = [0.50, 0.50, 0.00, 0.00]
	setH = [0.00, 0.00, 0.50, 0.50]
	print("    - Range Tests:")
	print("       -", setG, "vs", setG, "->", cosine_distance(setG, setG))
	print("       -", setF, "vs", setF, "->", cosine_distance(setF, setF))
	print("       -", setF, "vs", setG, "->", cosine_distance(setF, setG))
	print("       -", setH, "vs", setG, "->", cosine_distance(setH, setG))
	print("       -", setH, "vs", setF, "->", cosine_distance(setH, setF))

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
	# behaviour tests
	setF = [0.25, 0.25, 0.25, 0.25]
	setG = [0.50, 0.50, 0.00, 0.00]
	setH = [0.00, 0.00, 0.50, 0.50]
	print("    - Range Tests:")
	print("       -", setG, "vs", setG, "->", cosine_distance(setG, setG))
	print("       -", setF, "vs", setF, "->", cosine_distance(setF, setF))
	print("       -", setF, "vs", setG, "->", cosine_distance(setF, setG))
	print("       -", setH, "vs", setG, "->", cosine_distance(setH, setG))
	print("       -", setH, "vs", setF, "->", cosine_distance(setH, setF))

def testCDMeasure():
	"""
	@info Unit test to check accuracy of Cosine Distance Measure.
	"""
	# report
	print("\n - Testing CD Measure (Should Increase)")
	# initialize
	setA = [0.1, 0.2, 0.3, 0.4]
	setB = [0.1, 0.2, 0.4, 0.3]
	setC = [0.4, 0.3, 0.2, 0.1]
	setD = [0.3, 0.3, 0.3, 0.1]
	setE = [0.4, 0.3, 0.2, 0.1]
	# test cases
	print("    - Accuracy Tests:")
	print("       -", setA, "vs", setA, "->", cosine_distance(setA, setA))
	print("       -", setA, "vs", setB, "->", cosine_distance(setA, setB))
	print("       -", setA, "vs", setC, "->", cosine_distance(setA, setC))
	print("    - Symmetry Tests:")
	print("       -", setD, "vs", setE, "->", cosine_distance(setD, setE))
	print("       -", setE, "vs", setD, "->", cosine_distance(setE, setD))
	# behaviour tests
	setF = [0.25, 0.25, 0.25, 0.25]
	setG = [0.50, 0.50, 0.00, 0.00]
	setH = [0.00, 0.00, 0.50, 0.50]
	print("    - Range Tests:")
	print("       -", setG, "vs", setG, "->", cosine_distance(setG, setG))
	print("       -", setF, "vs", setF, "->", cosine_distance(setF, setF))
	print("       -", setF, "vs", setG, "->", cosine_distance(setF, setG))
	print("       -", setH, "vs", setG, "->", cosine_distance(setH, setG))
	print("       -", setH, "vs", setF, "->", cosine_distance(setH, setF))

def benchmarkJSMeasure():
	"""
	@info Check JS measure's behaviour for a 5-topic distinct synthetic log.
	"""

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t5-d.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

	###################
	## Test Accuracy ##
	###################

	# gather divergences
	print("\n - Gathering JS Divergences...")
	measures = calculateJSDivergences(corpus, dictionary, 10)
	# optimal
	optimalTopicCnt = findOptimalJSTopicCount(measures)
	print("\n - Optimal Topic Count", optimalTopicCnt)
	saveGraphY(measures, "Number of Topics", "JS Measure",
			   "JS Div (O=" + str(optimalTopicCnt) + ")",
			   "JS_topicDistributionBenchmark")

	# graph difference distributions
	print("\n - Graphing Distributions...")
	for i in range(0, 10):
		# test model
		model = generateLDAModel(corpus, dictionary, i+1)
		# print
		printTopicDistributions(model, dictionary)
		# visualize
		saveGraphOfTopics(extractTopicProbabilityDistributionFromModel(model, dictionary),
						  extractTopicWordsFromModel(model, dictionary),
						  "Topic Distributions (JSDiv=" + str(measures[i]) + ")",
						  "JS_topicDistributionBenchmark_" + str(i+1))

def benchmarkCDMeasure():
	"""
	@info Check CD measure's behaviour for a 5-topic distinct synthetic log.
	"""

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t5-d.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

	###################
	## Test Accuracy ##
	###################

	# gather divergences
	print("\n - Gathering CD Divergences...")
	measures = calculateCosineDistances(corpus, dictionary, 10)
	# optimal
	optimalTopicCnt = findOptimalCDTopicCount(measures)
	print("\n - Optimal Topic Count", optimalTopicCnt)
	saveGraphY(measures, "Number of Topics", "CD Measure",
			   "Cosine Distance (O=" + str(optimalTopicCnt) + ")",
			   "CS_topicDistributionBenchmark")

	# graph difference distributions
	print("\n - Graphing Distributions...")
	for i in range(0, 10):
		# test model
		model = generateLDAModel(corpus, dictionary, i+1)
		# print
		printTopicDistributions(model, dictionary)
		# visualize
		saveGraphOfTopics(extractTopicProbabilityDistributionFromModel(model, dictionary),
						  extractTopicWordsFromModel(model, dictionary),
						  "Topic Distributions (CDis=" + str(measures[i]) + ")",
						  "CD_topicDistributionBenchmark_" + str(i+1))

def benchmarkKLMeasure():
	"""
	@info Check JS measure's behaviour for a 5-topic distinct synthetic log.
	"""

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t5-d.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

	###################
	## Test Accuracy ##
	###################

	# gather divergences
	print("\n - Gathering KL Divergences...")
	measures = calculateKLDivergences(corpus, dictionary, 10)
	# optimal
	optimalTopicCntGen = findOptimalKLTopicCountGen(measures)
	optimalTopicCntNN = findOptimalKLTopicCountNN(measures)
	print("\n - Optimal Topic Count", optimalTopicCntGen,"or", optimalTopicCntNN)
	saveGraphY(measures, "Number of Topics", "KL (GEN) Measure",
			   "KL Divergence (O-BD=" + str(optimalTopicCntGen) + ", O-NN=" + str(optimalTopicCntNN) +")",
			   "KL_topicDistributionBenchmark")

	# graph difference distributions
	print("\n - Graphing Distributions...")
	for i in range(0, 10):
		# test model
		model = generateLDAModel(corpus, dictionary, i+1)
		# print
		printTopicDistributions(model, dictionary)
		# visualize
		saveGraphOfTopics(extractTopicProbabilityDistributionFromModel(model, dictionary),
						  extractTopicWordsFromModel(model, dictionary),
						  "Topic Distributions (KLDiv=" + str(measures[i]) + ")",
						  "KL_topicDistributionBenchmark_" + str(i+1))

###################
## Test Routines ##
###################

def runModelAccuracyTest():
	"""
	@info Examine the quality of models created with the correct topic count.
	"""

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t10-d.txt"
	optimalTopicCnt = 10
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

	###################
	## Test Accuracy ##
	###################

	for i in range(0, 5):
		# test model
		model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
		# print result
		printTopicDistributions(model, dictionary)
		# visualize
		saveGraphOfTopics(extractTopicProbabilityDistributionFromModel(model, dictionary),
							  extractTopicWordsFromModel(model, dictionary),
							  "Topic Distributions for a Distinct Synthetic Log",
							  "topicDistribution_d_" + str(optimalTopicCnt) + "_" + str(i))

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t10.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

	###################
	## Test Accuracy ##
	###################

	for i in range(0, 5):
		# test model
		model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
		# print result
		printTopicDistributions(model, dictionary)
		# visualize
		saveGraphOfTopics(extractTopicProbabilityDistributionFromModel(model, dictionary),
						  extractTopicWordsFromModel(model, dictionary),
						  "Topic Distributions for a Realistic Synthetic Log",
						  "topicDistribution_r_" + str(optimalTopicCnt) + "_" + str(i))

def runOptimalTopicNumberTest():
	"""
	@info Examine the topic predictions for all measures for a 10-topic log.
	"""

	###############
	## load data ##
	###############

	logFile = "data/testlogs/log-t10-d.txt"
	print("\n - Loading Log Data...")
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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
		saveGraphY(measuresKL, "Number of Topics", "KL Measure",
				   "KL Div (ONN=" + str(optimalKLTopicsNN) + ", OGen=" + str(optimalKLTopicsGen) + ")",
				   "Graph_" + str(x) + "_KL")

		# JS Measures
		print("\n - Testing JS topic ranges:")
		measuresJS = calculateJSDivergences(corpus, dictionary, LDA_MaxTopics)
		optimalJSTopics = findOptimalJSTopicCount(measuresJS)
		print("    = Optimal Topic Count:", optimalJSTopics)
		saveGraphY(measuresJS, "Number of Topics", "JS Measure",
				   "JS Div (O=" + str(optimalJSTopics) + ")",
				   "Graph_" + str(x) + "_JS")

		# CD Measures
		print("\n - Testing CD topic ranges:")
		measuresCD = calculateCosineDistances(corpus, dictionary, LDA_MaxTopics)
		optimalCDTopics = findOptimalCDTopicCount(measuresCD)
		print("    = Optimal Topic Count:", optimalCDTopics)
		saveGraphY(measuresCD, "Number of Topics", "CD Measure",
				   "JS Div (O=" + str(optimalCDTopics) + ")",
				   "Graph_" + str(x) + "_CD")

def runModellingTest():
	'''
    @info Test document classification of the LDA modelling process.
    '''

	###############
	## Test Data ##
	###############

	# initialize
	documents = ["desk desk desk",
				 "cat cat cat",
				 "computer computer computer",
				 "desk desk desk",
				 "cat cat cat",
				 "computer computer computer"]
	# report
	printDocuments(documents)

	############################
	## Prepare LDA Primitives ##
	############################

	# convert documents to token set
	tokens = generateTokens(documents)
	printTokens(tokens)

	# convert tokens to dictionary format (bow)
	dictionary = generateDictionary(tokens)
	printDictionary(dictionary)

	# generate corpus
	corpus = generateCorpus(dictionary, tokens)
	printCorpus(corpus)

	###################################
	## Determine Optimal Topic Count ##
	###################################

	# Calculates symmetric KL divergence.
	print("\n - Modelling data...")
	measures = calculateJSDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printMeasures(measures)

	# final optimal topic count
	optimalTopicCnt = findOptimalJSTopicCount(measures)
	print("\n - Optimal number of topics:", optimalTopicCnt)

	####################
	## LDA Model Test ##
	####################

	# build resulting optimal model
	print("\n - Resulting LDA Topics")
	LDA_Model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
	# print topics
	for i in LDA_Model.print_topics():
		print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

	# build actual optimal model
	print("\n - Actual LDA Topics")
	LDA_Model = generateLDAModel(corpus, dictionary, 3)
	# print topics
	for i in LDA_Model.print_topics():
		print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

	#############################################
	## Topic Distribution of New Document Test ##
	#############################################

	# seen pure document test
	testDoc = "computer computer computer computer"
	print("\n - Testing Unseen Document: '" + testDoc + "'")
	# determine topic distribution
	testDocTopics = classifyDocument(LDA_Model, testDoc)
	print("	- Topic Distribution:")
	for i in testDocTopics:
		print("	   - TOPIC_" + str(i[0]) + " : " + str(i[1]))

	# seen mixed document test
	testDoc = "computer cat computer cat"
	print("\n - Testing Unseen Document: '" + testDoc + "'")
	# determine topic distribution
	testDocTopics = classifyDocument(LDA_Model, testDoc)
	print("	- Topic Distribution:")
	for i in testDocTopics:
		print("	   - TOPIC_" + str(i[0]) + " : " + str(i[1]))

	# unseen document test
	testDoc = "dog dog dog"
	print("\n - Testing Unseen Document: '" + testDoc + "'")
	# determine topic distribution
	testDocTopics = classifyDocument(LDA_Model, testDoc)
	print("	- Topic Distribution:")
	for i in testDocTopics:
		print("	   - TOPIC_" + str(i[0]) + " : " + str(i[1]))

#################################
## Measure Evaluation Routines ##
#################################

def evaluateMeasurePerformanceDistinct():
	"""
	@info Evaluates the measure accuracies for a set of 5 distinct synthetic logs.
	"""

	############
	## Config ##
	############

	# imports
	global LDA_MaxTopics

	# initialize
	logIDs =    [2,  5,  10, 20, 40, 80]
	maxTopics = [16, 24, 32, 48, 88, 100]
	fileName = "data/Evaluation_Results_Distinct.txt"

	# make file
	file = open(fileName, "w")
	file.close()

	################
	## Evaluation ##
	################

	for pos in range(len(logIDs)):

		# setup
		ID = logIDs[pos]
		LDA_MaxTopics = maxTopics[pos]

		###############
		## load data ##
		###############

		logFile = "data/testlogs/log-t" + str(ID) + "-d.txt"
		print("\n - Loading Log Data...")
		print("	- location: " + logFile)
		logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

		# initialize
		runs = EVAL_runs
		optimalKLTopicsNN = []
		optimalKLTopicsGen = []
		optimalJSTopics = []
		optimalCDTopics = []

		# multiple runs
		for x in range(1, runs+1):

			# KL measures
			print("\n - Testing KL topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")

			measuresKL = calculateKLDivergences(corpus, dictionary, LDA_MaxTopics)
			optimalKLTopicsNN.append(findOptimalKLTopicCountNN(measuresKL))
			optimalKLTopicsGen.append(findOptimalKLTopicCountGen(measuresKL))

			# JS measure
			print("\n - Testing JS topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")
			measuresJS = calculateJSDivergences(corpus, dictionary, LDA_MaxTopics)
			optimalJSTopics.append(findOptimalJSTopicCount(measuresJS))

			# CDist measure
			print("\n - Testing CD topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")
			measuresCD = calculateCosineDistances(corpus, dictionary, LDA_MaxTopics)
			optimalCDTopics.append(findOptimalCDTopicCount(measuresCD))

		# output
		file = open(fileName, "a")
		# KL NN
		file.write("KL_NN," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalKLTopicsNN[i]))
		file.write("\n")
		# KL Gen
		file.write("KL_BD," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalKLTopicsGen[i]))
		file.write("\n")
		# JS
		file.write("JS," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalJSTopics[i]))
		file.write("\n")
		# CD
		file.write("CD," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalCDTopics[i]))
		file.write("\n")
		# done
		file.close()

def evaluateMeasurePerformanceRealistic():
	"""
	@info Evaluates the measure accuracies for a set of 5 realistic synthetic logs.
	"""

	############
	## Config ##
	############

	# imports
	global LDA_MaxTopics

	# initialize
	logIDs =    [2,  5,  10, 20, 40, 80]
	maxTopics = [16, 24, 32, 48, 88, 100]
	fileName = "data/Evaluation_Results_Realistic.txt"

	# make file
	file = open(fileName, "w")
	file.close()

	################
	## Evaluation ##
	################

	for pos in range(len(logIDs)):

		# setup
		ID = logIDs[pos]
		LDA_MaxTopics = maxTopics[pos]

		###############
		## load data ##
		###############

		logFile = "data/testlogs/log-t" + str(ID) + ".txt"
		print("\n - Loading Log Data...")
		print("	- location: " + logFile)
		logData = loadLogData(logFile, datetime.datetime(2017, 2, 1, 0, 0, 0))
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

		# initialize
		runs = EVAL_runs
		optimalKLTopicsNN = []
		optimalKLTopicsGen = []
		optimalJSTopics = []
		optimalCDTopics = []

		# multiple runs
		for x in range(1, runs+1):

			# KL measures
			print("\n - Testing KL topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")

			measuresKL = calculateKLDivergences(corpus, dictionary, LDA_MaxTopics)
			optimalKLTopicsNN.append(findOptimalKLTopicCountNN(measuresKL))
			optimalKLTopicsGen.append(findOptimalKLTopicCountGen(measuresKL))

			# JS measure
			print("\n - Testing JS topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")
			measuresJS = calculateJSDivergences(corpus, dictionary, LDA_MaxTopics)
			optimalJSTopics.append(findOptimalJSTopicCount(measuresJS))

			# CDist measure
			print("\n - Testing CD topic ranges: (run " + str(x) + " of " + str(runs) + ", file: " + str(ID) + ")")
			measuresCD = calculateCosineDistances(corpus, dictionary, LDA_MaxTopics)
			optimalCDTopics.append(findOptimalCDTopicCount(measuresCD))

		# output
		file = open(fileName, "a")
		# KL NN
		file.write("KL_NN," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalKLTopicsNN[i]))
		file.write("\n")
		# KL Gen
		file.write("KL_BD," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalKLTopicsGen[i]))
		file.write("\n")
		# JS
		file.write("JS," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalJSTopics[i]))
		file.write("\n")
		# CD
		file.write("CD," + str(ID))
		for i in range(runs):
			file.write("," + str(optimalCDTopics[i]))
		file.write("\n")
		# done
		file.close()

#################################
## Profile Evaluation Routines ##
#################################

def evaluatePredictionCapabilityForAllTimeScales():
	'''
	@info Tests prediction performance for all time scales.
	'''

	####################
	## Initialization ##
	####################

	# global imports
	global LDA_TimeScale
	originalTimeScale = LDA_TimeScale

	#####################
	## Run Evaluations ##
	#####################

	# evaluate daily profile
	for i in range(3):
		# choose scale
		LDA_TimeScale = i
		# run evaluation
		evaluatePredictionCapabilityForDailyUser()

	# evaluate weekly profile
	for i in range(3):
		# choose scale
		LDA_TimeScale = i
		# run evaluation
		evaluatePredictionCapabilityForWeeklyUser()

	#########################
	## Undo Config Changes ##
	#########################

	LDA_TimeScale = originalTimeScale

def evaluatePredictionCapabilityForDailyUser():
	'''
	@info Tests prediction performance for a single time scale.
	'''

	for run in range(3):

		####################
		## Initialization ##
		####################

		# initialize
		scores = []

		# load data
		print("\n - Loading Log Data...")
		logFile = "data/testlogs/log-user-daily.txt"
		print("	- location: " + logFile)
		FullLogData = loadLogData(logFile, datetime.datetime(2017, 1, 1, 0, 0, 0))
		print("	- items:", len(FullLogData))

		#####################
		## Prediction LOOP ##
		#####################

		for documentID in range(0, len(FullLogData)):

			############
			## Report ##
			############

			print("\n=== (DAILY USER) Testing item " + str(documentID + 1) + " of " + str(len(FullLogData)) + " for " + getTimeScaleAsString() + " scale, run " + str(run+1) + " of 3 ===")

			#########################
			## Generate Primitives ##
			#########################

			# load log data
			currentLogData = []
			documents = []
			for i in range(0, documentID):
				currentLogData.append(FullLogData[i])
				documents.append(FullLogData[i][1])

			# primitives
			tokens = generateTokens(documents)
			dictionary = generateDictionary(tokens)
			corpus = generateCorpus(dictionary, tokens)

			#####################
			## Build LDA Model ##
			#####################

			# final optimal topic count
			print("\n - Resulting LDA Topics:")
			measures = calculateJSDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
			optimalTopicCnt = findOptimalJSTopicCount(measures)
			print("\n - Optimal number of topics:", optimalTopicCnt)

			# make model
			LDA_Model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
			# print topics
			for i in LDA_Model.print_topics():
				print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

			#############################
			## Topics Time Association ##
			#############################

			# generate time factor data
			timeFactors = gatherTopicTimeFactors(LDA_Model, currentLogData)
			cleanedTimeFactors = cleanTimeFactors(timeFactors)
			# print time factors
			print("\n - Resulting Time Factors:")
			for i in cleanedTimeFactors:
				# print data
				print("	- TOPIC_" + str(i) + ": " + str(cleanedTimeFactors[i]), end="")
				# print string version
				print(" | [", end="")
				for j in range(0, len(cleanedTimeFactors[i])):
					print(timeFactorToTimeString(cleanedTimeFactors[i][j]), end="")
					if j < len(cleanedTimeFactors[i]) - 1:
						print(", ", end="")
				print("]")

			##############################
			## Try to predict next item ##
			##############################

			# find relevant topic
			print("\n - Trying to predict: " + FullLogData[documentID][1])
			timeFactor = FullLogData[documentID][0]
			relevantTopic = findMostRelevantTopic(timeFactor, cleanedTimeFactors)
			print("	= Relevant Topic:", relevantTopic)
			relevantTopicWords = getRelevantTopicKeyWords(LDA_Model, dictionary, relevantTopic)
			print("	= Relevant Topic Words:", relevantTopicWords)

			# initialize
			areAllWordsMatching = True
			targetDoc = (FullLogData[documentID][1]).lower()
			# check score
			for wordToken in relevantTopicWords:
				if not wordToken in targetDoc:
					areAllWordsMatching = False
					break
			# add score
			if areAllWordsMatching and len(relevantTopicWords) > 0:
				scores.append(1)
				print(" = Score awarded.")
			else:
				scores.append(0)
				print(" = NO Score awarded.")
			print(" = ", sum(scores) ,"/", documentID + 1)

		##################
		## Save Results ##
		##################

		# report
		print("\n - Saving results...")
		# graph form
		saveGraphY(scores, "Document ID", "Cumulative Score", "Daily User Predication Performance, TimeScale=" + getTimeScaleAsString(), "graph_prediction_d_" + getTimeScaleAsString() + "_" + str(run))
		# file form
		file = open("data/Evaluation_Results_Prediction_d_" + getTimeScaleAsString() + "_" + str(run) +".txt", "w")
		file.write(getTimeScaleAsString() + ",")
		for x in range(len(scores)):
			file.write(str(scores[x]))
			if x < len(scores) - 1:
				file.write(",")
		# done
		file.close()

def evaluatePredictionCapabilityForWeeklyUser():
	'''
	@info Tests prediction performance for a single time scale.
	'''

	for run in range(3):

		####################
		## Initialization ##
		####################

		# initialize
		scores = []

		# load data
		print("\n - Loading Log Data...")
		logFile = "data/testlogs/log-user-weekly-2.txt"
		print("	- location: " + logFile)
		FullLogData = loadLogData(logFile, datetime.datetime(2017, 1, 1, 0, 0, 0))
		print("	- items:", len(FullLogData))

		#####################
		## Prediction LOOP ##
		#####################

		for documentID in range(0, len(FullLogData)):

			############
			## Report ##
			############

			print("\n=== (WEEKLY USER) Testing item " + str(documentID + 1) + " of " + str(len(FullLogData)) + " for " + getTimeScaleAsString() + " scale, run " + str(run+1) + " of 3 ===")

			#########################
			## Generate Primitives ##
			#########################

			# load log data
			currentLogData = []
			documents = []
			for i in range(0, documentID):
				currentLogData.append(FullLogData[i])
				documents.append(FullLogData[i][1])

			# primitives
			tokens = generateTokens(documents)
			dictionary = generateDictionary(tokens)
			corpus = generateCorpus(dictionary, tokens)

			#####################
			## Build LDA Model ##
			#####################

			# final optimal topic count
			print("\n - Resulting LDA Topics:")
			measures = calculateJSDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
			optimalTopicCnt = findOptimalJSTopicCount(measures)
			print("\n - Optimal number of topics:", optimalTopicCnt)

			# make model
			LDA_Model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
			# print topics
			for i in LDA_Model.print_topics():
				print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

			#############################
			## Topics Time Association ##
			#############################

			# generate time factor data
			timeFactors = gatherTopicTimeFactors(LDA_Model, currentLogData)
			cleanedTimeFactors = cleanTimeFactors(timeFactors)
			# print time factors
			print("\n - Resulting Time Factors:")
			for i in cleanedTimeFactors:
				# print data
				print("	- TOPIC_" + str(i) + ": " + str(cleanedTimeFactors[i]), end="")
				# print string version
				print(" | [", end="")
				for j in range(0, len(cleanedTimeFactors[i])):
					print(timeFactorToTimeString(cleanedTimeFactors[i][j]), end="")
					if j < len(cleanedTimeFactors[i]) - 1:
						print(", ", end="")
				print("]")

			##############################
			## Try to predict next item ##
			##############################

			# find relevant topic
			print("\n - Trying to predict: " + FullLogData[documentID][1])
			timeFactor = FullLogData[documentID][0]
			relevantTopic = findMostRelevantTopic(timeFactor, cleanedTimeFactors)
			print("	= Relevant Topic:", relevantTopic)
			relevantTopicWords = getRelevantTopicKeyWords(LDA_Model, dictionary, relevantTopic)
			print("	= Relevant Topic Words:", relevantTopicWords)

			# initialize
			areAllWordsMatching = True
			targetDoc = (FullLogData[documentID][1]).lower()
			# check score
			for wordToken in relevantTopicWords:
				if not wordToken in targetDoc:
					areAllWordsMatching = False
					break
			# add score
			if areAllWordsMatching and len(relevantTopicWords) > 0:
				scores.append(1)
				print(" = Score awarded.")
			else:
				scores.append(0)
				print(" = NO Score awarded.")
			print(" = ", sum(scores), "/", documentID + 1)

		##################
		## Save Results ##
		##################

		# report
		print("\n - Saving results...")
		# graph form
		saveGraphY(scores, "Document ID", "Cumulative Score", "Weekly User Predication Performance, TimeScale=" + getTimeScaleAsString(), "graph_prediction_w_" + getTimeScaleAsString() + "_" + str(run))
		# file form
		file = open("data/Evaluation_Results_Prediction_w_" + getTimeScaleAsString() +"_" + str(run) + ".txt", "w")
		file.write(getTimeScaleAsString() + ",")
		for x in range(len(scores)):
			file.write(str(scores[x]))
			if x < len(scores) - 1:
				file.write(",")
		# done
		file.close()

#########################
## Program Entry Point ##
#########################

if __name__ == '__main__':
	main()