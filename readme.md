##########
## Info ##
##########

	Created by Tashiv Sewpersad for use in the TimeRank project. All aspects
	of this project are licensed under the GNU General Public License, which
	can be found at <http://www.gnu.org/licenses/>.

##############
## Overview ##
##############

	This project contains three software artefacts, namely the user modelling
	segment of the TimeRank Porject, a tool for generating user data logs and
	a Chrome Extension template which demonstrates native messaging using 
	python 3. The core software artefact is the user modelling program contained
	in the Profiler.py and Profiler_Grapher.py files. This program has two modes 
	of operation and this is specified in the following section.
	
#######################
## Using the project ##
#######################

	The project's user modelling tool has two modes of operation. The first is 
	the testing mode which has a command line type interface. This presents 
	the user with a range of tests to execute in order to test various aspects
	of the user modelling process. This was used to generate the data for the
	project's write-up. This mode can be activated using:
	
		python3 Profiler.py -b
		
	Calling the program without any parameters activates the second mode of 
	operation. This default mode is intended for use by the Chrome Extension
	part of the TimeRank project. This mode processes the accompanying 
	'file-log.txt' in order to produce a user profile which will be used
	in re-ranking. This mode is activate using:
	
		python3 Profiler.py
		
#######################
## Project Structure ##
#######################

/data = all test logs are stored here. Additionally all data produced by the
        testing mode is also stored in here.
		
/formats = a set of documents which specify the user profile format and the
           way in which a time factor is calculated.
		   
/tools = contains the log generator and the chrome extension template. I.e. these
         are the secondary artefacts mentioned above. Also the rule files used to
		 generate the test logs can be found within this directory as well.
		 