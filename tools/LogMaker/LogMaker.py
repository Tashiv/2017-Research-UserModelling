#########
# Notes #
#######################################################################
#                                                                     #
# Limitations:                                                        #
#  [1] - Log items may have the same times which is unrealistic.      #
#                                                                     #
#######################################################################

#############
## Imports ##
#############

from dateutil.relativedelta import relativedelta
from datetime import datetime
import random
import sys

###################
## Configuration ##
###################

fConfig = "config.txt"
fLog = "log.txt"
fDelimiter = " -- "

######################
## Global Variables ##
######################

fStartDate = datetime.now()
fEndDate = datetime.now()
fRules = []

##########
## Main ##
##########

def main():
    # header
    print("[Log Generator]")
    # global Variables
    global fConfig
    # process parameters
    if  len(sys.argv) == 3:
        if sys.argv[1] == "-c":
            fConfig = sys.argv[2]
    # load config
    loadConfig()
    # report configuration
    print("   - Log Range:",fStartDate,"->",fEndDate)
    print("   - Number of Rules:", len(fRules))
    # build Log
    buildLog()
    # done
    print("[Exited]")

#############
## Methods ##
#############

def loadConfig():
    # report
    print(" - Loading '" + fConfig + "' configuration file...")
    # global variables
    global fStartDate
    global fEndDate
    # initialize
    currentRule = None
    # open file
    file = open(fConfig, "r")
    # read lines
    for line in file:
        # clean line
        line = line.strip()
        # process line
        if line == "":
            continue
        elif line[0] == '#':
            continue
        elif line[0:6] == "start ":
            # extract date
            lineData = line.split(' ')
            dateData = lineData[1].split('-')
            timeData = lineData[2].split(':')
            # set date
            fStartDate = datetime(int(dateData[0]), int(dateData[1]), int(dateData[2]), int(timeData[0]), int(timeData[1]), int(timeData[2]))
        elif line[0:4] == "end ":
            # extract date
            lineData = line.split(' ')
            dateData = lineData[1].split('-')
            timeData = lineData[2].split(':')
            # set date
            fEndDate = datetime(int(dateData[0]), int(dateData[1]), int(dateData[2]), int(timeData[0]), int(timeData[1]), int(timeData[2]))
        elif line[0:3] == "-- ":
            currentRule.data.append(line[3::])
        else:
            currentRule = LogBuildRule(line)
            fRules.append(currentRule)
    # close file
    file.close()
    # sanity check
    if (fEndDate < fStartDate):
        fStartDate, fEndDate = fEndDate, fStartDate

def buildLog():
    # report
    print(" - Building Log...")
    # initialize
    logItems = []
    ruleID = 0
    # process each build rule
    for buildRule in fRules:
        # report rule
        ruleID+=1
        print("    - Processing rule " + str(ruleID) + '...')
        # initialize
        stopDate = fStartDate + relativedelta(seconds=buildRule.timeStop.second,
                                            minutes=buildRule.timeStop.minute,
                                            hours=buildRule.timeStop.hour,
                                            days=buildRule.timeStop.day,
                                            months=buildRule.timeStop.month,
                                            years=buildRule.timeStop.year)
        currentDate = fStartDate

        # apply initial offset
        currentDate += relativedelta(seconds=buildRule.timeStart.second,
                                            minutes=buildRule.timeStart.minute,
                                            hours=buildRule.timeStart.hour,
                                            days=buildRule.timeStart.day,
                                            months=buildRule.timeStart.month,
                                            years=buildRule.timeStart.year)
        # keep making log items until final date
        while (currentDate < fEndDate) and (currentDate < stopDate):
            # calculate probability of occuring
            randomValue = random.random()
            if randomValue < buildRule.sessionProbability:
                # make N number of items to mimic a session
                for i in range(0, random.randint(buildRule.minSessionOccurences, buildRule.maxSessionOccurences)):
                    # apply variance to time
                    variedDate = currentDate + relativedelta(seconds=random.randint(-1*buildRule.timeVariance.second,buildRule.timeVariance.second),
                                                                minutes=random.randint(-1*buildRule.timeVariance.minute,buildRule.timeVariance.minute),
                                                                hours=random.randint(-1*buildRule.timeVariance.hour,buildRule.timeVariance.hour),
                                                                days=random.randint(-1*buildRule.timeVariance.day,buildRule.timeVariance.day),
                                                                months=random.randint(-1*buildRule.timeVariance.month,buildRule.timeVariance.month),
                                                                years=random.randint(-1*buildRule.timeVariance.year,buildRule.timeVariance.year))
                    # make item
                    logItems.append(LogItem(variedDate, buildRule.getRandomDataItem()))
            # increment date
            currentDate += relativedelta(seconds=buildRule.timeInterval.second,
                                            minutes=buildRule.timeInterval.minute,
                                            hours=buildRule.timeInterval.hour,
                                            days=buildRule.timeInterval.day,
                                            months=buildRule.timeInterval.month,
                                            years=buildRule.timeInterval.year)
    # sort log items
    print("    - sorting " + str(len(logItems)) + " items...")
    logItems.sort()
    # output to file
    print("    - saving to " + fLog + "...")
    file = open(fLog, "w")
    for item in logItems:
        file.write(str(item) + '\n')
    file.close()

#############
## Classes ##
#############

class TimeSet:
    ## variables ##
    second = 0
    minute = 0
    hour = 0
    day = 0
    month = 0
    year = 0

    ## constructor ##
    def __init__(self, timeString):
        # process data
        timeData = timeString.split(',')
        # assign data
        self.second = int(timeData[0])
        self.minute = int(timeData[1])
        self.hour = int(timeData[2])
        self.day = int(timeData[3])
        self.month = int(timeData[4])
        self.year = int(timeData[5])

class LogBuildRule:
    ## VARIABLES ##
    timeStart = None
    timeStop = None
    timeInterval = None
    timeVariance = None
    sessionProbability = 0
    minSessionOccurences = 0
    maxSessionOccurences = 0
    data = None

    ## CONSTRUCTOR ##
    def __init__(self, logString):
        # process data
        logData = logString.split(fDelimiter)
        # assign rule times
        self.timeStart = TimeSet(logData[0])
        self.timeStop = TimeSet(logData[1])
        self.timeInterval = TimeSet(logData[2])
        self.timeVariance = TimeSet(logData[3])
        self.sessionProbability = float(logData[4])
        self.minSessionOccurences = int(logData[5])
        self.maxSessionOccurences = int(logData[6])
        # store rule data items
        self.data = []
        for i in range(7, len(logData)):
            self.data.append(logData[i])

    ## METHODS ##
    def getRandomDataItem(self):
        # choose random item
        index = random.randint(0, len(self.data)-1)
        # provide it
        return self.data[index]

class LogItem:
    ## VARIABLES ##
    time = datetime.now
    data = "<NONE>"

    ## CONSTRUCTOR ##
    def __init__(self, time, data):
        # assign data
        self.time = time
        self.data = data

    ## METHODS ##
    def __lt__(self, other):
         return self.time < other.time

    def __str__(self):
        return self.time.strftime("%Y,%m,%d,%H,%M,%S") + "_#_" + self.data

#########################
## Program Entry Point ##
#########################

if __name__ == "__main__":
    main()
