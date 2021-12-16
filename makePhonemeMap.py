from globalVars import DIR_PATH, SCRIPT_PATH, REMOVE_NUMBERS
import os

def getPhoneList():
	# This function gets the unique values of all phonemes used in the alignments files and returns them into an array

	phoneList = []

	outputFolder = DIR_PATH + '/MontrealAlignments/'

	for speaker in os.listdir(outputFolder):
		for session in os.listdir(f"{outputFolder}/{speaker}"):
			for filename in os.listdir(f"{outputFolder}/{speaker}/{session}"):
				if 'phones' in filename:
					file = open(f"{outputFolder}/{speaker}/{session}/{filename}",'r')
					lines = file.readlines()
					file.close()

					for line in lines:
						if line.strip():
							phoneme = line.split(' ')[-1]
							phoneme = phoneme.replace('\n','')

							if REMOVE_NUMBERS: # When this option is true, the numbers are removed from phoneme names and no distinctions are done for the same phoneme
								phoneme = ''.join(char for char in phoneme if not char.isdigit())
							
							if not phoneme in phoneList:
								phoneList.append(phoneme)

	return sorted(phoneList)

def makeMixedLabels(phoneList):
	# This function calculates all posible combinations between different phonemes and adds them to the phoneme array

	completePhoneList = phoneList[:]

	for i in phoneList:
		for j in phoneList:
			if i != j:
				completePhoneList.append(f"{i}+{j}")
				completePhoneList.append(f"{i}-{j}")

	return completePhoneList

def writePhoneMap(phoneList):
	# This function writes the phoneme array to a text file, asigning a number to each phoneme

	with open(f"{SCRIPT_PATH}/phoneMap",'w+') as file:
		file.write(f"no_label -1\n")
		for i in range(len(phoneList)):
			file.write(f"{phoneList[i]} {i}\n")

def main():

	phoneList = getPhoneList()

	phoneList = makeMixedLabels(phoneList)

	writePhoneMap(phoneList)

if __name__ == '__main__':
	main()
