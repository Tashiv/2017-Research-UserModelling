#############
## Imports ##
#############

import time

#############
## Methods ##
#############

def main():
	# add to log
	file = open("test.txt",'w')
	file.write("[START]\n")
	file.close()
	for i in range(0 , 30):
		time.sleep(1)
		file = open("test.txt",'a')
		file.write(str(i) + "\n")
		file.close()
	file = open("test.txt",'a')
	file.write("[END]\n")
	file.close()

###################
## Program Entry ##
###################

if __name__ == "__main__":
	main()
	print("Ended...")
