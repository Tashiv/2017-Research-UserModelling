#############
## Imports ##
#############

import time

######################
## Global Variables ##
######################

profileFilename = "file-profile.txt"

#############
## Methods ##
#############

def main():
	# add to log
	file = open(profileFilename,'w')
	file.write("[START]\n")
	file.close()
	for i in range(0 , 30):
		time.sleep(1)
		file = open(profileFilename,'a')
		file.write(str(i) + "\n")
		file.close()
	file = open(profileFilename,'a')
	file.write("[END]\n")
	file.close()

###################
## Program Entry ##
###################

if __name__ == "__main__":
	print("[Profiler]")
	print(" - Running...")
	main()
	print("[ENDED]")
