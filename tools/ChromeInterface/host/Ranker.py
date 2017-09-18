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

logFilename = "file-log.txt"
errorLogFilename = "file-error-logger.txt"

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
	# process input
	if (messageDictionary["action"] == "rerank"):
		# rerank links
		result = reRank(messageDictionary)
		# prepare response
		response = {}
		response["action"] = "rerank"
		response["payload"] = result
		# send response
		sendMessage(response)

def reRank(resultString):
	# initialize
	items = []
	counter = 0
	# loop on result item
	for item in resultString.split("_$_"):
		# process elements
		elements = item.split("_#_")
		title = elements[0]
		link = elements[1]
		blurb = elements[2]
		# store
		items.append(link)
	# "re-rank" items
	result = ""
	for i in range(len(item)-1,0,-1):
		# add item
		result += str(i)
		# delimter
		if (i > 0):
			result += ","
	# done
	return result

###################
## Program Entry ##
###################

if __name__ == "__main__":
	try:
		processInput()
	except Exception as e:
		file = open(errorLogFilename,'a')
		file.write(str(e) + "\n")
		file.close()
