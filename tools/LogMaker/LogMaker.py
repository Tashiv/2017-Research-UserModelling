#############
## Imports ##
#############

import datetime
import dateutil

###################
## Configuration ##
###################

fConfig = "config.txt"
fLog = "log.txt"
fStartDate = datetime.datetime.now()
fEndDate = datetime.datetime.now()
fRules = []

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
    ## variables ##
    timeInterval = TimeSet("0,0,0,0,0,0")
    timeOffset = TimeSet("0,0,0,0,0,0")
    data = "<NONE>"

    ## constructor ##
    def __init__(self, logString):
        # process data
        logData = logString.split(" -- ")
        # assign data
        self.timeOffset = TimeSet(logData[0])
        self.timeInterval = TimeSet(logData[1])
        self.data = logData[2]

class LogItem:
    ## variables ##
    time = datetime.datetime.now
    data = "<NONE>"

    ## constructor ##
    def __init__(self, time, data):
        # assign data
        self.time = time
        self.data = data

    ## sorting interface ##
    def __lt__(self, other):
         return self.time < other.time

    ## output interface ##
    def __str__(self):
        return str(self.time) + " " + self.data

#############
## Methods ##
#############

def loadConfig():
    # global variables
    global fStartDate
    global fEndDate
    # initialize
    iLogDays = 0
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
        elif line[0:5] == "date ":
            # extract date
            lineData = line.split(' ')
            dateData = lineData[1].split('-')
            # set date
            fStartDate = datetime.datetime(int(dateData[0]), int(dateData[1]), int(dateData[2]), fStartDate.hour, fStartDate.minute, fStartDate.second)
        elif line[0:5] == "time ":
            # extract time
            lineData = line.split(' ')
            timeData = lineData[1].split(':')
            # set date
            fStartDate = datetime.datetime(fStartDate.year, fStartDate.month, fStartDate.day, int(timeData[0]), int(timeData[1]), int(timeData[2]))
        elif line[0:5] == "days ":
            # store number of days
            iLogDays = line.split(" ")[1]
        else:
            fRules.append(LogBuildRule(line))

    # close file
    file.close()
    # set end date
    fEndDate = fStartDate + datetime.timedelta(days=int(iLogDays))

def buildLog():
    # initialize
    print(" - Building Log...")
    logItems = []
    iRuleCnt = 0
    # process each build rule
    for buildRule in fRules:
        # initialize
        fCurrentDate = fStartDate
        iRuleCnt+=1
        # apply offset
        fCurrentDate += datetime.timedelta(seconds=buildRule.timeOffset.second,
                                            minutes=buildRule.timeOffset.minute,
                                            hours=buildRule.timeOffset.hour,
                                            days=(buildRule.timeOffset.day + buildRule.timeOffset.month*30 + buildRule.timeOffset.year*356))
        # keep making log items until final date
        while fCurrentDate < fEndDate:
            # make item
            logItems.append(LogItem(fCurrentDate, buildRule.data))
            # increment date
            fCurrentDate += datetime.timedelta(seconds=buildRule.timeInterval.second,
                                                minutes=buildRule.timeInterval.minute,
                                                hours=buildRule.timeInterval.hour,
                                                days=(buildRule.timeInterval.day + buildRule.timeInterval.month*30 + buildRule.timeInterval.year*356))
        # report
        print("    - processed rule", iRuleCnt, '.')

    # sort log items
    print("    - sorting " + str(len(logItems)) + " items...")
    logItems.sort()
    # output to file
    print("    - saving to " + fLog + "...")
    file = open(fLog, "w")
    for item in logItems:
        file.write(str(item) + '\n')
    file.close()
    # done
    print("    - done.")

##########
## Main ##
##########

def main():
    # load config
    loadConfig()
    # report range
    print("[Log Generator]")
    print(" - Log Range:",fStartDate,"->",fEndDate)
    print(" - Number of Rules:", len(fRules))
    # build Log
    buildLog()
    # done
    print("[Exited]")

#########################
## Program Entry Point ##
#########################

if __name__ == "__main__":
    main()
