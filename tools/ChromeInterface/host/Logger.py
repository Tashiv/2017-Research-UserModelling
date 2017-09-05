# SOURCE: https://chromium.googlesource.com/chromium/src/+/master/chrome/common/extensions/docs/examples/api/nativeMessaging/host/native-messaging-example-host
# Modified by 'russian_koopa': https://www.reddit.com/r/learnpython/comments/4yo4fn/i_cant_write_a_python_35_script_that_works_with/
# Adapted for TimeRank.

#############
## Imports ##
#############

import json
import struct
import sys

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
	# process input (PLACEHOLDER)
	file = open("log.txt",'a')
	file.write("[NEW-RUN]\n")
	file.write(" - Stored:" + str(messageDictionary) + "\n")
	file.close
	# send response
	sendMessage({"Some Important Numbers" : "1 2 3 0.5"})

###################
## Program Entry ##
###################

if __name__ == "__main__":
	try:
		processInput()
	except Exception as e:
		file = open("TimeRank-log.txt",'a')
		file.write("[NEW-RUN]\n")
		file.write(str(e) + "\n")
		file.close
