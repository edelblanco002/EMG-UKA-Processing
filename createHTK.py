from globalVars import DIR_PATH, MARK_CONTEXT_PHONEMES
import os
import textgrids as tg # praat-textgrids library

def createFolders(speaker,session):
	# This function creates the subfolder tree for saving the .lab files

	try:
		os.makedirs(f"{DIR_PATH}/MontrealAlignments/{speaker}/{session}")
	except OSError as error:
		print(error)

def markContext(phones: list):
	"""This function marks the context phonemes of the "ata(CV)ta" type utterances.
		The following possibilities are considered:
			- The phoneme list could start with the 'sil' phone mark or not.
			- The phoneme list could end with the 'sp' phone mark or not.
			- Some utterances have an additional phoneme just before the (CV).
	"""

	if phones[0].text == 'sil':
		offset1 = 1
	elif phones[0].text == 'a':
		offset1 = 0
	else:
		print(f"ERROR: unexpected first phoneme: {phones[0].text}")
		raise ValueError

	if phones[-1].text == 'sp':
		additionalLength = 1
	elif phones[-1].text == 'a':
		additionalLength = 0
	else:
		print(f"ERROR: unexpected last phoneme: {phones[-1].text}")
		raise ValueError

	phones[0 + offset1].text = 'a0'
	phones[1 + offset1].text = 't0'
	phones[2 + offset1].text = 'a1'

	if len(phones) == 8 + additionalLength + offset1:
		phones[3 + offset1].text += '0'
		offset2 = 1
	elif len(phones) == 7 + additionalLength + offset1:
		offset2 = 0
	else:
		print(f"ERROR: unexpected number of phones")
		raise ValueError

	phones[5 + offset1 + offset2].text = 't1'
	phones[6 + offset1 + offset2].text = 'a2'

	return phones

def writeHTKFiles(speaker,session,outputFolder,filename):
	# This function writes the .lab files in HTK format.
	# It creates one file for word alignment and another one for phoneme alignment

	try:
		grid = tg.TextGrid(f"{DIR_PATH}/{outputFolder}/{speaker}/{session}/{filename}")
	except:
		print("Error when opening ",filename)
		raise OSError

	words = grid['words']
	phones = grid['phones']

	utt = filename.split('-')[-1]
	utt = utt.replace('.TextGrid','')

	wordFile = f"{DIR_PATH}/MontrealAlignments/{speaker}/{session}/words_{speaker}_{session}_{utt}.lab"
	phoneFile = wordFile.replace('words','phones') # phones_{speaker}_{session}_{utt}.lab

	# Write .lab file for word alignment
	with open(wordFile, 'w+') as file:
		for interval in words:
			label = interval.text.transcode()
			if label == '':
				label = '_' # Silences are labeled with '_' symbol
			start = interval.xmin
			end = interval.xmax
			file.write(f"{start} {end} {label}\n")

	if MARK_CONTEXT_PHONEMES:
		phones = markContext(phones)

	# Write .lab file for phoneme alignment
	with open(phoneFile,'w+') as file:
		for interval in phones:
			label = interval.text
			start = interval.xmin
			end = interval.xmax
			file.write(f"{start} {end} {label}\n")

def main():

	outputFolders = ['audibleOutput','whisperedOutput']

	for outputFolder in outputFolders:
		for speaker in os.listdir(f"{DIR_PATH}/{outputFolder}"):
			for session in os.listdir(f"{DIR_PATH}/{outputFolder}/{speaker}"):
				# Create a new folder for each speaker and session in the audibleOutput/whisperedOutput folder
				createFolders(speaker,session)
				# Create an HTK file for every utterance in the session
				for filename in os.listdir(f"{DIR_PATH}/{outputFolder}/{speaker}/{session}"):
					writeHTKFiles(speaker,session,outputFolder,filename)

if __name__ == '__main__':
	main()
