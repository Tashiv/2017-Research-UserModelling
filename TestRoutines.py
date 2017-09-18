from Profiler import *
import sys

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
## Evaluation Routines ##
#########################

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
		printMeasures(KL_Divergences)
		# find optimal topic count
		optimalTopicCnt = findOptimalKLTopicCountNN(KL_Divergences)
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
	saveGraphY(parameterValues, topicCounts, "LDA Topic Runs", "Topic Count", "LDA_TopicRuns")
	saveGraphY(parameterValues, buildTimes, "LDA Topic Runs", "Build Time", "LDA_TopicRuns_Times")

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
	printMeasures(KL_Divergences)

	# final optimal topic count
	optimalTopicCnt = findOptimalKLTopicCountNN(KL_Divergences)
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
	saveUserProfile(LDA_Model, cleanedTimeFactors, "data/userProfile.txt")

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
	printMeasures(KL_Divergences)

	# final optimal topic count
	optimalTopicCnt = findOptimalKLTopicCountNN(KL_Divergences)
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