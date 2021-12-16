from globalVars import CORPUS, DIR_PATH, N_CHANNELS
import numpy as np
import os
from scipy.io import wavfile
import struct

def createFolders(session, speaker):
    try:
        os.makedirs(DIR_PATH + '/audioSync/' + speaker + '/' + session)
    except OSError as error:
        print(error)
    
    try:
        os.makedirs(DIR_PATH + '/emgSync/' + speaker + '/' + session)
    except OSError as error:
        print(error)

def createSyncAudio(speaker,session,filename,audioStart,audioEnd):
    # This function loads the audio file, removes the right channel, which has a synchronization signal,
    # removes the offset and saves the samples into a numpy array file

    """if CORPUS == 'Pilot Study':
        audioPrefix = 'audio_'
    elif CORPUS == 'EMG-UKA':"""
    audioPrefix = 'a_'
    """else:
        print("ERROR: Wrong CORPUS name in globalVars.py.")
        raise ValueError"""

    audioFs, audioData = wavfile.read(f"{DIR_PATH}/audio/{speaker}/{session}/{audioPrefix}{filename}.wav")
    audioData = audioData[:,0] # Get only left channel
    audioData = audioData[audioStart:audioEnd] # Remove offset

    wavfile.write(f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{filename}.wav",audioFs,audioData)
    np.save(f"{DIR_PATH}/audioSync/{speaker}/{session}/a_{filename}.npy", np.array(audioData))

def createSyncEMG(speaker, session, filename, emgChannels, emgStart, emgEnd):
    # This function reads the emg data from a binary .adc file
    # (Samples are signed short integers, little-indian)
    # Then rearranges the 7 channels into a matrix,
    # removes the offset from all signals
    # and saves the resulting matrix into a .npy file

    nBytes = 2
    dataType = '<h' # little-indian, short signed integer
    
    decoded = []
    
    """if CORPUS == 'Pilot Study':
        emgPrefix = 'emg_' + str(emgChannels).zfill(2) + 'ch'
    elif CORPUS == 'EMG-UKA':"""
    emgPrefix = 'e' + str(emgChannels).zfill(2)
    """else:
        print("ERROR: Wrong CORPUS name in globalVars.py.")
        raise ValueError"""
    
    # Read binary data
    with open(DIR_PATH + '/emg/' + speaker + '/' + session + '/' + emgPrefix + '_' + filename + '.adc','rb') as f:
        while True:
            try:
                buf = f.read(nBytes)
                decoded.append(struct.unpack_from(dataType,buf))
            except:
                break
    
    decodedArray = np.array(decoded)
    decodedMatrix = decodedArray.reshape(int(len(decodedArray)/emgChannels),emgChannels) # Rearrange channels
    decodedMatrix = decodedMatrix[emgStart:emgEnd,:] # Remove offset
    np.save(DIR_PATH + '/emgSync/' + speaker + '/' + session + '/e' + str(emgChannels).zfill(2) + '_' + filename + '.npy',decodedMatrix)

def readSynchronism(speaker,session,filename):
    # This function gets the offset parameters from the offset file
    # The offset file is a text file that has two lines:
    # In the first line are the start and the end offset for the audio signal
    # In the second line are the start and the end offset for the emg signals
    # The offsets are given in samples

    fid = open(DIR_PATH + '/offset/' + speaker + '/' + session + '/offset_' + filename + '.txt')
    lines = fid.readlines()
    fid.close()
    
    columns = lines[0].split(' ')
    audioStart = int(columns[0])
    audioEnd = int(columns[1].replace('\n',''))
    
    columns = lines[1].split(' ')
    emgStart = int(columns[0])
    emgEnd = int(columns[1].replace('\n',''))
    return audioEnd, audioStart, emgEnd, emgStart

# dirname: Root folder
def main():
    # Main function
    # This script loads the offset file off each sample
    # and crops its corresponding audio and emg signals to remove the offset
    # The cropped files are saved into audioSync and emgSync folders

    emgChannels = N_CHANNELS # Used emg channels

    for speaker in os.listdir(DIR_PATH + '/audio'):
        print("Processing speaker n. ",speaker,':')
        
        for session in os.listdir(DIR_PATH + '/audio/' + speaker):
            print("  Processing session ",session,':')

            createFolders(session, speaker) 

            for file in os.listdir(DIR_PATH + '/audio/' + speaker + '/' + session):

                """if CORPUS == 'Pilot Study':
                    filename = file[6:-4] # Removes 'audio_' and '.wav' exception in order to get the basename
                elif CORPUS == 'EMG-UKA':"""
                filename = file[2:-4] # Removes 'a_' and '.wav' exception in order to get the basename
                """else:
                    print("ERROR: Wrong CORPUS name in globalVars.py.")
                    raise ValueError"""

                print("     Processing: ",filename)

                audioEnd, audioStart, emgEnd, emgStart = readSynchronism(speaker,session,filename)

                createSyncAudio(speaker,session,filename,audioStart,audioEnd)

                createSyncEMG(speaker, session, filename, emgChannels, emgStart, emgEnd)

if __name__ == "__main__":
    main()
