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
import sys
import datetime
import time
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
LDA_SlidingWindow = 7											# How old the log is allowed to be in days
LDA_Passes = 1
LDA_Iterations = 50
LDA_EvalEvery = None											# Don't evaluate model perplexity, takes too much time.
LDA_Alpha = 'symmetric'
LDA_Eta = None
LDA_Minimum_Probability = 0.01
LDA_MaxTopics = 16
LDA_TopicRuns = 10												# Number of times each topic KL_Divergence is evaluated, reduces noise
LDA_MaxThreads = 8
LDA_MinimumTimeFactorDifference = 0.01							# Used in combining similar time factors
LDA_MinKLDivergenceDifference = 0.05

## Tools
fTokenizer = RegexpTokenizer(r'\w+')							# Regex for matching words made up of alphanumeric and underscore characters
fStemmer = PorterStemmer()										# Tool for stemming tokens
fStopWordsList = get_stop_words('en')							# English stop words list

##########
## Main ##
##########

def main():
	# header
	print("[Topic Modeller]")
	# testing
	if len(sys.argv) != 2:
		buildProfile()
	else:
		if sys.argv[1] == "-t1":
			runModellingTest()
		elif sys.argv[1] == "-t2":
			runProfileGeneratorTest()
		elif sys.argv[1] == "-b":
			runBenchmark()
		elif sys.argv[1] == "-p":
			runParameterTester()
		elif sys.argv[1] == "-j":
			testJSD()
		else:
			printArguments()
	# done
	print("\n[DONE]")

def printArguments():
	print("\n Arguments:")
	print("  -t1 : Runs basic modelling test.")
	print("  -t2 : Runs a profile generating test.")
	print("  -b  : Generates KL Divergence results.")
	print("  -p  : Tests the modeller for a range of parameter values.")
	print("  -j  : Tests JSD.")

#############################
## Chrome Interface Plugin ##
#############################

def buildProfile():
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
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)

	#####################
	## Build LDA Model ##
	#####################

	# calculate divergences
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printKLDivergences(KL_Divergences)
	# final optimal topic count
	optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
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
	createUserProfile(LDA_Model, cleanedTimeFactors, logFilename)

#########################
## Evaluation Routines ##
#########################

def testJSD():
	# load data
	print("\n - Loading Log Data...")
	logFile = "data/testlog-t8-d.txt"
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017,2,1,0,0,0))
	logItemCnt = len(logData)
	print("	- items:", logItemCnt)
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
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)
	# multiple runs
	for x in range(3):
		# calculate JS Divergences
		print("\n - Testing topic ranges:")
		JS_Divergences = []
		runs = 1
		for i in range(1, LDA_MaxTopics + 1):
			# report
			print("	- Testing " + str(i) + " of " + str(LDA_MaxTopics) + ": ", end = "")
			# calculate
			temp = []
			for run in range(0, runs):
				temp.append(calculatedJSDivergence(corpus, dictionary, i))
			# store
			result = sum(temp) / len(temp)
			JS_Divergences.append(result)
			print(str(result))
		# output
		saveJSDivergencesGraph(JS_Divergences, "JS_Divergences_" + str(x))

def runBenchmark():
	# load data
	print("\n - Loading Log Data...")
	logFile = "data/testlog-t4-d.txt"
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017,2,1,0,0,0))
	logItemCnt = len(logData)
	print("	- items:", logItemCnt)
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
		if (currentItem % 250 == 0):
			print("	- Processed " + str(currentItem) + " of " + str(logItemCnt))
	# Generate Primatives
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)
	# benchmark KL divergences
	for run in range(1, 4):
		# find divergences
		print("\n - Building Model...")
		KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
		saveKLDivergencesGraph(KL_Divergences, "KL_Divergences_"
												+ str(LDA_TopicRuns) + "TopicRuns_"
												+ str(LDA_Iterations) + "Iterations_"
												+ str(LDA_Passes) + "Passes_"
												+ str(run))
		printKLDivergences(KL_Divergences)
		# find optimal topic count
		optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
		print("\n - Optimal number of topics:", optimalTopicCnt)
		# show topics
		LDA_Model = generateLDAModel(corpus, dictionary, optimalTopicCnt)
		print("\n - Resulting LDA Topics")
		for i in LDA_Model.print_topics():
			print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

def runParameterTester():
	###################
	## Load Log Data ##
	###################

	# load data
	print("\n - Loading Log Data...")
	logFile = "data/testlog-t10-d.txt"
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime(2017,2,1,0,0,0))
	logItemCnt = len(logData)
	print("	- items:", logItemCnt)

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
		if (currentItem % 250 == 0):
			print("	- Processed " + str(currentItem) + " of " + str(logItemCnt))

	# prepare primatives
	print("\n - Saving KL Divergence Profile as graph...")
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)

	#####################
	## Build LDA Model ##
	#####################

	# initialize
	parameterValues = []
	topicCounts = []
	buildTimes = []
	# [PART 1/3] : settings
	global LDA_TopicRuns
	minParameter = 1
	maxParameter = 30
	parameterSteps = 1
	runs = 1
	# run Tests
	for parameterValue in range(minParameter, maxParameter, parameterSteps):
		# report
		print("\n*** parameter set to " + str(parameterValue) + " of " + str(maxParameter) + " (step=" + str(parameterSteps) + ") ***")
		# initialize
		tempTopicCounts = []
		tempBuildTimes = []
		# average runs
		for j in range(1, runs+1):
			# [PART 2/3] : set parameter value
			LDA_TopicRuns = parameterValue
			# build resulting optimal model
			timer = int(round(time.time() * 1000))
			LDA_Model = generateLDAModelFromPrimatives(corpus, dictionary)
			timer = (int(round(time.time() * 1000)) - timer) / 1000
			print("	- Took " + str(timer) + " seconds to do run " + str(j) + " of " + str(runs))
			# store values
			tempTopicCounts.append(len(LDA_Model.print_topics()))
			tempBuildTimes.append(timer)
		# log values
		topicCounts.append(sum(tempTopicCounts) / len(tempTopicCounts))
		buildTimes.append(sum(tempBuildTimes) / len(tempBuildTimes))
		parameterValues.append(parameterValue)
	# [PART 3/3] : graph results
	print("\n - Saving " + str(len(parameterValues)) + " data points in graph form...")
	saveGeneralGraph(parameterValues, topicCounts, "LDA Topic Runs", "Topic Count", "LDA_TopicRuns")
	saveGeneralGraph(parameterValues, buildTimes, "LDA Topic Runs", "Build Time", "LDA_TopicRuns_Times")

def runProfileGeneratorTest():
	'''
	@info Generates a basic user profile from a log.
	'''
	###################
	## Load Log Data ##
	###################

	# load data
	print("\n - Loading Log Data...")
	logFile = "data/testlog.txt"
	print("	- location: " + logFile)
	logData = loadLogData(logFile, datetime.datetime.now())
	print("	- items:", len(logData))

	# extract documents
	documents = []
	for i in logData:
		documents.append(i[1])
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)

	#####################
	## Build LDA Model ##
	#####################

	# build resulting optimal model
	LDA_Model = generateLDAModelFromDocuments(documents)
	# print topics
	print("\n - Resulting LDA Topics:")
	for i in LDA_Model.print_topics():
		print("	- TOPIC_" + str(i[0]) + ": " + str(i[1]))

	# calculate divergences
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printKLDivergences(KL_Divergences)
	saveKLDivergencesGraph(KL_Divergences, "KL_Divergences")

	# final optimal topic count
	optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
	print("\n - Optimal number of topics:", optimalTopicCnt)

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

	print("\n - saving user profile to data/userProfile.txt...")
	createUserProfile(LDA_Model, cleanedTimeFactors, "data/userProfile.txt")

	############################
	## Finding Relevant topic ##
	############################

	print("\n - Please enter a time (H:M) or 'q' to quit:\n	> ", end="")
	line = input()
	while line != "q":
		# process line
		line = line.strip().split(':')
		# make into a time factor
		timestamp = datetime.datetime.now()
		timestamp = datetime.datetime(timestamp.year, timestamp.month, timestamp.day, int(line[0]), int(line[1]), timestamp.second)
		timeFactor = calculateTimeFactor(timestamp)
		# find relevant topic
		print("	= Relevant topic is TOPIC_" + str(findMostRelevantTopic(timeFactor, cleanedTimeFactors)))
		# get a new line
		print("\n - Please enter a time (H:M) or 'q' to quit:\n	> ", end="")
		line = input()

	##########
	## Done ##
	##########

	print("\n[EXITED]")

def runModellingTest():
	'''
	@info Generates a trivial model and reports each stage for testing purposes.
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
	## Prepare LDA Primatives ##
	############################

	# convert documents to token set
	tokens = generateTokens(documents)
	printTokens(tokens)

	# convert tokens to dictionary format (bow)
	dictionary = generateDictionary(tokens, True)
	printDictionary(dictionary)

	# generate corpus
	corpus = generateCorpus(dictionary, tokens)
	printCorpus(corpus)

	###################################
	## Determine Optimal Topic Count ##
	###################################

	# Caluculates symmetric KL divergence.
	print("\n - Modelling data...")
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	printKLDivergences(KL_Divergences)
	saveKLDivergencesGraph(KL_Divergences, "KL_Divergences")

	# final optimal topic count
	optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
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

	##########
	## Done ##
	##########

	print("\n[EXITED]")

##################################
## LDA Modelling Helper Methods ##
##################################

def generateTokens(documents):
	'''
	@info Converts an array of documents to tokens and cleans them using stop words and stemming.
	'''
	# initialize
	tokens_final = []
	# loop through document list
	for i in documents:
		# add tokens to final list
		tokens_final.append(generateTokensFromString(i))
	# Done
	return tokens_final

def generateTokensFromString(string):
	# convert to lower case and tokenize document string
	tokens_cleaned = fTokenizer.tokenize(string.lower())
	# remove stop words from tokens
	tokens_stopped = [i for i in tokens_cleaned if not i in fStopWordsList]
	# stem tokens
	tokens_stemmed = [fStemmer.stem(i) for i in tokens_stopped]
	# done
	return tokens_stemmed

def generateDictionary(tokens, mustClean):
	'''
	@info Generates a dictionary from a set of tokens and cleans the dictionary if required.
	'''
	# generate dictionary
	dictionary = corpora.Dictionary(tokens)
	# apply filtering
	if mustClean:
		# FILTER 1: filter out uncommon tokens (appear only once)
		unique_ids = [token_id for token_id, frequency in dictionary.iteritems() if frequency == 1]
		dictionary.filter_tokens(unique_ids)
		# FILTER 2: filter out common tokens (appear in more than 5 documents)
		## dictionary.filter_extremes(no_above=5, keep_n=100000)
		# CLEAN: Reassign ids to 'fill gaps'
		dictionary.compactify()
	# done
	return dictionary

def generateCorpus(dictionary, tokens):
	# generate corpus
	corpus = [dictionary.doc2bow(token) for token in tokens]
	# done
	return corpus

def generateLDAModel(corpus, dictionary, topicCount):
	return models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topicCount,
										iterations=LDA_Iterations, passes=LDA_Passes, eval_every=LDA_EvalEvery,
										alpha=LDA_Alpha, eta=LDA_Eta, minimum_probability=LDA_Minimum_Probability)

def generateLDAModelFromDocuments(documents):
	# report
	print("\n - Building Model...")
	# prepare LDA primatives
	tokens = generateTokens(documents)
	dictionary = generateDictionary(tokens, True)
	corpus = generateCorpus(dictionary, tokens)
	# determine optimal topic count
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
	print("	- Optimal topic count is " + str(optimalTopicCnt) + ".")
	# build model
	return generateLDAModel(corpus, dictionary, optimalTopicCnt)

def generateLDAModelFromPrimatives(corpus, dictionary):
	# report
	print("\n - Building Model...")
	# determine optimal topic count
	KL_Divergences = calculateKLDivergences(corpus, dictionary, max_topics=LDA_MaxTopics)
	optimalTopicCnt = findKLDivergenceDip(KL_Divergences)
	print("	- Optimal topic count is " + str(optimalTopicCnt) + ".")
	# build model
	return generateLDAModel(corpus, dictionary, optimalTopicCnt)

#################################
## LDA Model Interface Methods ##
#################################

def classifyDocument(LDA_Model, document):
	# convert to token format
	tokens = generateTokensFromString(document)
	bagOfWords = LDA_Model.id2word.doc2bow(tokens)
	# determine topics
	return LDA_Model.get_document_topics(bagOfWords, minimum_probability=0)

###########################
## LDA Reporting Methods ##
###########################

def printDocuments(documents):
	# heading
	print("\n - Raw Documents:")
	# output
	for i in range(0, len(documents)):
		print("	- [DOC_" + str(i) + "] : " + documents[i])

def printTokens(tokens):
	# heading
	print("\n - Generated Document Tokens:")
	# output
	for i in range(0, len(tokens)):
		print("	- [DOC_" + str(i) + "] : " + str(tokens[i]))

def printDictionary(dictionary):
	# heading
	print("\n - Generated Dictionary:")
	# output
	for i in dictionary:
		print("	- [DICT_" + str(i) + "] : " + dictionary[i])

def printCorpus(corpus):
	# heading
	print("\n - Generated Corpus:")
	# output
	for i in range(0, len(corpus)):
		print("	- [DOC_" + str(i) + "] : " + str(corpus[i]))

def printKLDivergences(KL_Divergences):
	# heading
	print("\n - Topic Symmetric KL Divergences:")
	# output
	for i in range(0, len(KL_Divergences)):
		print("	- [" + str(i+1) + "_TOPIC(S)] : " + str(KL_Divergences[i]))

###############################
## Jensen-shannon divergence ##
###############################

def jensen_shannon_divergence(p,q):
	# initialize
	result = 0
	average = numpy.zeros(len(p))
	weight = 0.5
	for i in range(0, len(p)):
		average[i] = weight * p[i] + (1 - weight) * q[i]
		result = (weight * symmetric_kl_divergence(numpy.array(p), average)) + ((1 - weight) * symmetric_kl_divergence(numpy.array(q), average))
	return (result / numpy.sqrt(2 * math.log(2)))

def jensen_shannon_divergenceV1(x,y):
	'''
	@author: jonathanfriedman
	'''
	x = numpy.array(x)
	y = numpy.array(y)
	d1 = x*numpy.log2(2*x/(x+y))
	d2 = y*numpy.log2(2*y/(x+y))
	d1[numpy.isnan(d1)] = 0
	d2[numpy.isnan(d2)] = 0
	d = numpy.sum(d1+d2)
	return d

def calculatedJSDivergence(corpus, dictionary, number_of_topics):
	# initialize
	lda_model = generateLDAModel(corpus, dictionary, number_of_topics)
	k_factor = 1
	if number_of_topics > 1:
		k_factor = 1 / (number_of_topics * (number_of_topics - 1))
	# build topics list
	topics = []
	for i in range(0, number_of_topics):
		# sort items
		items = []
		for item in lda_model.show_topics(num_topics=-1, num_words=len(dictionary), formatted=False)[i][1]:
			items.append(item)
		# items.sort(key=lambda tup: tup[0])
		# format
		topic = []
		for item in items:
			topic.append(item[1])
		# done
		topics.append(topic)
	# compare resulting topics
	k_sum = 0
	for i in range(0, number_of_topics):
		for j in range(0, number_of_topics):
			if i >= j:
				continue
			m = topics[i]
			n = topics[j]
			k_sum += jensen_shannon_divergence(m,n)
	# multiply factor
	result = k_factor * k_sum
	# done
	return result

###########################
## KL Divergence Methods ##
###########################

def symmetric_kl_divergenceV1(p, q):
		""" Compute KL divergence of two vectors, K(p || q)."""
		return sum(p[x] * math.log((p[x]) / (q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

def symmetric_kl_divergence(p, q):
	'''
	@info Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	'''
	return numpy.sum([stats.entropy(p, q), stats.entropy(q, p)])

def calculateKLDivergence(corpus, dictionary, number_of_topics):
	# Generates corpus length vectors.
	corpus_length_vector = numpy.array([sum(frequency for _, frequency in document) for document in corpus])
	# Instanciates LDA.
	lda = generateLDAModel(corpus, dictionary, number_of_topics)
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
	document_topic_vector = document_topic_vector
	document_topic_norm   = numpy.linalg.norm(corpus_length_vector)
	document_topic_vector = document_topic_vector / document_topic_norm
	# calculate KL divergence
	return symmetric_kl_divergence(document_word_vector, document_topic_vector)

def calculateKLDivergences(corpus, dictionary, max_topics=1):
	# sanity check
	token_count = len(dictionary)
	if max_topics > token_count:
		# report
		print("	- Warning: Max_topics is more than number of tokens, using " + str(token_count) + " instead of " + str(max_topics) + " as max topics.")
		# update
		max_topics = token_count
	# use correct threading mode
	if (LDA_MaxThreads == 1):
		return calculateKLDivergencesST(corpus, dictionary, max_topics)
	else:
		return calculateKLDivergencesMT(corpus, dictionary, max_topics)

def calculateKLDivergencesST(corpus, dictionary, max_topics=1):
	"""
	@info   Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA, Modified by Tashiv Sewpersad
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	# initialize
	result =  [None] * max_topics
	# calculate KL divergence
	print("	- Trying various topic counts: (max=" + str(max_topics) + ", SINGLE Thread Mode)")
	for i in range(0, max_topics, 1):
		# calculate divergence
		KLWorker(corpus, dictionary, i+1, result)
	# done
	return result

def calculateKLDivergencesMT(corpus, dictionary, max_topics=1):
	"""
	@info   Caluculates symmetric Kullback-Leibler divergence.
	@author Shingo OKAWA, Modified by Tashiv Sewpersad
	@source https://gist.github.com/shingoOKAWA/b8f92cc0f6f0183767dc
	"""
	# initialize
	result =  multiprocessing.Array(ctypes.c_double, max_topics)
	# calculate KL divergence
	print("	- Trying various topic counts: (max=" + str(max_topics) + ", MULTI-Threads=" + str(LDA_MaxThreads) + ")")
	jobs = []
	for i in range(0, max_topics, 1):
		# Create new threads
		worker = multiprocessing.Process(target=KLWorker, args=(corpus, dictionary, i+1, result,))
		worker.start()
		jobs.append(worker)
		# batch processing
		if (len(jobs) >= LDA_MaxThreads):
			for j in jobs:
				j.join()
			jobs = []
	# wait for remaining threads to finish
	for j in jobs:
		j.join()
	# done
	return result

def findKLDivergenceDip(KL_Divergences):
	'''
	@info Finds the dip in the divergence values which indicates optimal number of topics
	'''
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
			if (KL_Divergences[i] > KL_Divergences[i-1] - LDA_MinKLDivergenceDifference):
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

#########################
## Time Factor Methods ##
#########################

def calculateTimeFactor(timestamp):
	return abs((timestamp.hour*60+timestamp.minute)/1440)

def calculateWeighting(timeFactor1, timeFactor2):
	return 1 - min(abs(timeFactor1 - timeFactor2), 1 - abs(timeFactor1 - timeFactor2))

def timeFactorToTimeString(timeFactor):
	# extract parts
	hour = int((timeFactor * 1440) / 60)
	minute = int((timeFactor * 1440) % 60)
	# make string
	return str(hour) + ":" + str(minute)

def findMostRelevantTopic(currentTimeFactor, modelTimeFactors):
	# initialize
	bestTopicID = -1
	bestWeighting = -1
	# find relevant topic
	for topicID in modelTimeFactors:
		for timeFactor in modelTimeFactors[topicID]:
			weighting = calculateWeighting(currentTimeFactor, timeFactor)
			if (weighting > bestWeighting):
				bestTopicID = topicID
				bestWeighting = weighting
	# done
	return bestTopicID

def gatherTopicTimeFactors(LDA_Model, logData):
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
	# initialize
	cleanedTimeFactors = dict()
	LDA_MinimumTimeFactorDifference = 0.01
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
			if foundSimilarElement == False:
				cleanedTimeFactors[i].append(timeFactor)
	# done
	return cleanedTimeFactors

##############
## File i/o ##
##############

def saveGeneralGraph(xValues, yValues, xAxisName, yAxisName, graphName):
	'''
	@info Saves a graph of values to disk.
	'''
	# setup graph
	pyplot.close('all')
	pyplot.plot(xValues, yValues, color="red")
	pyplot.ylabel(yAxisName, color="black")
	pyplot.xlabel(xAxisName, color="black")
	# save graph
	pyplot.savefig('data/' + graphName + '.png', facecolor='white', edgecolor='white', bbox_inches='tight')

def saveKLDivergencesGraph(KL_Divergences, graphName):
	'''
	@info Saves a graph of KL Divergences to disk.
	'''
	# generate x axis values
	xAxisValues = []
	for i in range(1, len(KL_Divergences)+1):
		xAxisValues.append(i)
	# setup graph
	pyplot.close('all')
	pyplot.plot(xAxisValues, KL_Divergences, color="red")
	pyplot.ylabel('Symmetric KL Divergence', color="black")
	pyplot.xlabel('Number of Topics', color="black")
	# save graph
	pyplot.savefig('data/' + graphName + '.png', facecolor='white', edgecolor='white', bbox_inches='tight')

def saveJSDivergencesGraph(JS_Divergences, graphName):
	'''
	@info Saves a graph of JS Divergences to disk.
	'''
	# generate x axis values
	xAxisValues = []
	for i in range(1, len(JS_Divergences)+1):
		xAxisValues.append(i)
	# setup graph
	pyplot.close('all')
	pyplot.plot(xAxisValues, JS_Divergences, color="red")
	pyplot.ylabel('JS Divergence', color="black")
	pyplot.xlabel('Number of Topics', color="black")
	# save graph
	pyplot.savefig('data/' + graphName + '.png', facecolor='white', edgecolor='white', bbox_inches='tight')

def loadLogData(filename, referenceDate):
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
		if ((referenceDate - timestamp).days >= LDA_SlidingWindow):
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

def createUserProfile(LDA_Model, timeFactors, fileName):
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
				if (i < len(topicKeywords)-1):
					file.write(", ")
			file.write("]\n")
	# close file
	file.close()

##########################
## Multiprocess Workers ##
##########################

def KLWorker(corpus, dictionary, number_of_topics, result):
	# report
	print("	   - Trying a topic count of " + str(number_of_topics) + "...")
	# average values
	tempValues = []
	for j in range(0, LDA_TopicRuns):
		# calculate KL divergence
		tempValues.append(calculateKLDivergence(corpus, dictionary, number_of_topics))
	# add smallest result
	# avg: result[number_of_topics-1] = sum(tempValues) / len(tempValues)
	result[number_of_topics-1] = min(tempValues)

#########################
## Program Entry Point ##
#########################

if __name__ == '__main__':
	main()
