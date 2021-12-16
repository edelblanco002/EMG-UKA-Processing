from globalVars import DIR_PATH, CORPUS
import os
from shutil import copyfile

def getAudibleList():
	# This function looks into the subset definitions to find which files corresponds to audible utterances

	audibleList = []
	audibleListingFiles = ['train.audible', 'test.audible']

	for filename in audibleListingFiles:
		file = open(DIR_PATH + '/Subsets/' + filename,'r')
		lines = file.readlines()
		file.close()

		for line in lines:
			if line.strip():
				line = line.replace('\n','')
				columns = line.split(' ')
				columns = columns[1:] # First position corresponds to speaker/session identifier
				for element in columns: # Each utterance reference
					if element != '':
						element = element.replace('emg_','') # Remove the prefix
						audibleList.append(element)

	return audibleList

def getSilentList():
	# This function looks into the subset definitions to find which files corresponds to silent utterances

	silentList = []
	silentListingFiles = ['train.silent', 'test.silent']

	for filename in silentListingFiles:
		file = open(DIR_PATH + '/Subsets/' + filename,'r')
		lines = file.readlines()
		file.close()

		for line in lines:
			if line.strip():
				line = line.replace('\n','')
				columns = line.split(' ')
				columns = columns[1:] # First position corresponds to speaker/session identifier
				for element in columns: # Each utterance reference
					if element != '':
						element = element.replace('emg_','') # Remove the prefix
						silentList.append(element)

	return silentList

def getWhisperedList():
	# This function looks into the subset definitions to find which files corresponds to whispered utterances

	whisperedList = []
	whisperedListingFiles = ['train.whispered', 'test.whispered']

	for filename in whisperedListingFiles:
		file = open(DIR_PATH + '/Subsets/' + filename,'r')
		lines = file.readlines()
		file.close()

		for line in lines:
			if line.strip():
				line = line.replace('\n','')
				columns = line.split(' ')
				columns = columns[1:] # First position corresponds to speaker/session identifier
				for element in columns: # Each utterance reference
					if element != '':
						element = element.replace('emg_','') # Remove the prefix
						whisperedList.append(element)

	return whisperedList

def makeAudibleCorpus(audibleList):
	# This function copies the audible utterances into the audibleCorpus folder

	for element in audibleList:
		[speaker,session,utt] = element.split('-')
		#if CORPUS == 'EMG-UKA':
		srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_{session}_{utt}"
		"""elif CORPUS == 'Pilot Study':
			srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_Audible{session[-1]}_{utt}"
		else:
			print("Wrong CORPUS value in globalVars.py")
			raise ValueError"""
		dstBase = srcBase.replace('audioSync','audibleCorpus')
		for ext in ['.wav','.npy','.lab']:
			try:
				copyfile(srcBase + ext,dstBase + ext)
			except OSError as error:
				print(error)

def makeCorporaStructure():
	# This function creates the three new folders that will hold each kind of utterance
	# and their corresponding folder three

	for speaker in os.listdir(DIR_PATH + '/audioSync'):
		for session in os.listdir(DIR_PATH + '/audioSync/' + speaker):
			try:
				os.makedirs(DIR_PATH + '/audibleCorpus/' + speaker + '/' + session)
			except OSError as error:
				print(error)

			try:
				os.makedirs(DIR_PATH + '/whisperedCorpus/' + speaker + '/' + session)
			except OSError as error:
				print(error)

			try:
				os.makedirs(DIR_PATH + '/silentCorpus/' + speaker + '/' + session)
			except OSError as error:
				print(error)

def makeSilentCorpus(silentList):
	# This function copies the silent utterances into the silentCorpus folder

	for element in silentList:
		[speaker,session,utt] = element.split('-')
		#if CORPUS == 'EMG-UKA':
		srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_{session}_{utt}"
		"""elif CORPUS == 'Pilot Study':
			srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_Silent{session[-1]}_{utt}"
		else:
			print("Wrong CORPUS value in globalVars.py")
			raise ValueError"""
		dstBase = srcBase.replace('audioSync','silentCorpus')
		for ext in ['.wav','.npy','.lab']:
			try:
				copyfile(srcBase + ext,dstBase + ext)
			except OSError as error:
				print(error)

def makeWhisperedCorpus(whisperedList):
	# This function copies the whispered utterances into the whisperedCorpus folder

	for element in whisperedList:
		[speaker,session,utt] = element.split('-')
		#if CORPUS == 'EMG-UKA':
		srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_{session}_{utt}"
		"""elif CORPUS == 'Pilot Study':
			srcBase = f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{speaker}_Whispered{session[-1]}_{utt}"
		else:
			print("Wrong CORPUS value in globalVars.py")
			raise ValueError"""
		dstBase = srcBase.replace('audioSync','whisperedCorpus')
		for ext in ['.wav','.npy','.lab']:
			try:
				copyfile(srcBase + ext,dstBase + ext)
			except OSError as error:
				print(error)

def main():

	audibleList = getAudibleList()
	whisperedList = getWhisperedList()
	silentList = getSilentList()

	makeCorporaStructure()
	makeAudibleCorpus(audibleList)
	makeWhisperedCorpus(whisperedList)
	makeSilentCorpus(silentList)

if __name__ == '__main__':
	main()
