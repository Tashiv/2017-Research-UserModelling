# SOURCE: https://chromium.googlesource.com/chromium/src/+/master/chrome/common/extensions/docs/examples/api/nativeMessaging/host/native-messaging-example-host
# Modified by 'russian_koopa': https://www.reddit.com/r/learnpython/comments/4yo4fn/i_cant_write_a_python_35_script_that_works_with/
# Adapted for TimeRank.

#############
## Imports ##
#############

import json
import struct
import sys
import subprocess

######################
## Global Variables ##
######################

counterFilename = "counter.txt"
logFilename = "log.txt"
errorLogFilename = "error-logger.txt"

####################
## Chrome Methods ##
####################

def sendMessage(messageDictionary):
	# convert to json and encode
	messageJson = json.dumps(messageDictionary, separators=(",", ":"))
	messageJsonEncoded = messageJson.encode("utf-8")
	# transmit message size
	sys.stdout.buffer.write(struct.pack("i", len(messageJsonEncoded)))
	# transmit message
	sys.stdout.buffer.write(messageJsonEncoded)

def readMessage():
	# read message length
    messageSizeBytes = sys.stdin.buffer.read(4)
    messageSize = struct.unpack("i", messageSizeBytes)[0]
	# read and decode message
    messageDecoded = sys.stdin.buffer.read(messageSize).decode("utf-8")
    # convert to json dictionary
    messageDictionary = json.loads(messageDecoded)
    # Returns the dictionary.
    return messageDictionary

######################
## TimeRank Methods ##
######################

def processInput():
	# read in message data
	messageDictionary = readMessage()
	# add to log
	file = open(logFilename,'a')
	for value in messageDictionary:
		file.write(str(messageDictionary[value]) + "\n")
	file.close
	# call profiler
	if (incrementAndGetLogCounter() >= 10):
		# reset counter
		resetCounter()
		# activate profiler
		process = subprocess.Popen("python Profiler.py", stdout=subprocess.PIPE)

def incrementAndGetLogCounter():
	# initialize
	counter = -1
	try:
		# read counter
		file = open(counterFilename,'r')
		counter = int(file.read())
		file.close()
	except:
		counter = 0
	# increment
	counter += 1
	# write
	file = open(counterFilename,'w')
	file.write(str(counter))
	file.close()
	# done
	return counter

def resetCounter():
	# write
	file = open(counterFilename,'w')
	file.write("0")
	file.close()

###################
## Program Entry ##
###################

if __name__ == "__main__":
	try:
		processInput()
	except Exception as e:
		file = open(errorLogFilename,'w')
		file.write(str(e) + "\n")
		file.close
