from bar import printProgressBar
from scipy.fftpack.pseudo_diffs import shift
from globalVars import DIR_PATH, N_CHANNELS, REMOVE_NUMBERS, SCRIPT_PATH
from globalVars import FS, FRAME_SHIFT, FRAME_SIZE, FEATURES_PER_FRAME, N_FEATURES, STACKING_WIDTH
from globalVars import AUDIO_FRAME_SIZE, AUDIO_FRAME_SHIFT, N_FILTERS, N_COEF
import math
import numpy as np
import os
import python_speech_features as psf
from scipy.io import wavfile
import sys

def buildPhoneDict():
    # This function reads the phoneMap file and creates a dictionary to translate the phoneme names to numbers

    file = open(f"{SCRIPT_PATH}/phoneMap")
    lines = file.readlines()
    file.close()

    phoneDict = {}

    for line in lines:
        if line.strip():
            line = line.replace('\n','')
            [key,value] = line.split(' ')
            phoneDict[key] = int(value)

    return phoneDict

def calculateAverage(sig):
    # This function returns the average of each frame,
    # by using a window of L points

    averagedSignal = [0]*len(sig)
    L = 9 # Number of points of the window (must be an odd number)
    # The actual sample for which the average is being calculated is going to be 
    # in the middle of the window, so the first sample is taken n samples before
    n = int((L-1)/2)
    
    for i in range(len(sig)):
        start = i - n
        end = i + n
        wdw = [0]*L
    
        # If the window exceeds the signal, missing values are left as 0
        if start < 0:
            wdw[start*-1:] = sig[0:end+1]
        elif end >= len(sig):
            wdw[:len(sig)-(end+1)] = sig[start:]
        else:
            wdw[:] = sig[start:end+1]
    
        averagedSignal[i] = sum(wdw)/L # An L pointed average

    return averagedSignal

def createFolders(session, speaker, folderName):
    try:
        os.makedirs(DIR_PATH + '/' + folderName + '/' + speaker + '/' + session)
    except OSError as error:
        pass
        #print(error)

def getLabelSegments(filename,srcFolderName,srcExt):
    # This function reads the HTK file related to the file and returns an array with the label names and their corresponding end boundaries

    path = '/'.join(filename.split('/')[:-1])
    labFilename = filename.split('/')[-1]

    path = path.replace(srcFolderName,'MontrealAlignments')

    if labFilename.startswith('a_'): # If is the case of an audio file
        labFilename = labFilename[1:]
        labFilename = 'phones' + labFilename
    else:
        labFilename = labFilename.replace('e' + str(N_CHANNELS).zfill(2) ,'phones')
    
    labFilename = labFilename.replace(srcExt,'.lab')

    labFilePath = f"{path}/{labFilename}"

    if os.path.isfile(labFilePath):
        labelSegments = []

        file = open(labFilePath,'r')
        lines = file.readlines()
        file.close()

        for line in lines:
            if line.strip():
                line = line.replace('\n','')
                columns = line.split(' ')
                end = float(columns[1])
                phone = columns[2]

                if REMOVE_NUMBERS: # If this option is activated, the numbers are removed from the name of the labels and no distinction is done between phonemes of the same class
                    phone = ''.join(char for char in phone if not char.isdigit())

                labelSegments.append([end,phone])

        return labelSegments

    else:
        return [[10000.0,'no_label']] # If there is no transcription file, as for silent utterances, every frames are going to be labeled as 'no_label'

def getPhoneLabel(j,L,step,fs,labelSegments):
    # This function assigns a label name to the frame
    
    # In order to avoid possible incosistencies due to the rounding, those values are recalculated from the actual sample amount
    frameSize = float(L/fs)
    frameShift = float(step/fs)

    # Calculate the position of the start, middle and end points of the frame in seconds
    start = j*frameShift
    middle = start+frameSize/2
    end = start+frameSize

    triphonemeFlag = False # It's user to detect possible triphonemes

    i = 0
    startPhoneme = labelSegments[0][1]
    # The start phoneme assigned is the first label whose upper boundary is greater than the start time mark of the frame
    while i < len(labelSegments) and start > labelSegments[i][0]:
        startPhoneme = labelSegments[i+1][1]
        i+=1


    i = 0
    middlePhoneme = labelSegments[i][1]
    # The middle phoneme assigned is the first label whose upper boundary is greater than the middle time mark of the frame
    while (i + 1) < len(labelSegments) and middle > labelSegments[i][0]:
        middlePhoneme = labelSegments[i+1][1]
        i+=1

    i = 0
    endPhoneme = labelSegments[i][1]
    # The end phoneme assigned is the first label whose upper boundary is greater than the end time mark of the frame
    while (i + 1) < len(labelSegments) and end > labelSegments[i][0]:
        endPhoneme = labelSegments[i+1][1]
        i+=1

    label = middlePhoneme

    # Guess whether is a transition frame
    if startPhoneme != middlePhoneme:
        label = f"{startPhoneme}-{middlePhoneme}"
        triphonemeFlag = True

    if endPhoneme != middlePhoneme:
        if triphonemeFlag: # If all start, middle and end points lies into different labels, there is a triphoneme
            # Nothing is done but launch a warning
            print("WARNING: There is a triphoneme")
        else:
            label = f"{middlePhoneme}+{endPhoneme}"

    return label

def stackingFilter(TD0, i, j, k, nFeatures,nFrames):
    # This function applies the stacking filter to a frame
    # The stacking filter takes the features of the k previous frames, the actual frame and the k following frames,
    # and puts them together into a single array

    td15 = [0]*(2*k+1)*nFeatures

    for ind in range(0,2*k + 1):
        jj = j - k + ind
        
        if jj < 0 or jj >= nFrames: # If the range of the stacking filter exceeds the frame position,
            td15[ind*nFeatures:(ind+1)*nFeatures] = [np.nan]*nFeatures # leave to 0 the features corresponding to those positions
        else:
            td15[ind*nFeatures:(ind+1)*nFeatures] = TD0[i,jj,:]

    return td15

def zeroCrossingCount(frame):
    # This function counts the zero crossings into the frame

    signs = np.sign(frame)
    signs[signs == 0] = -1
    return len(np.where(np.diff(signs))[0])

def extractFeatues():
    fs = FS
    frameSize = FRAME_SIZE # In ms
    frameShift = FRAME_SHIFT # In ms
    k = STACKING_WIDTH # Width of stacking filter
    nFeatures = N_FEATURES # Number of calculated features    

    phoneDict = buildPhoneDict()

    for speaker in os.listdir(DIR_PATH + '/emgSync'):
        print("Processing speaker n. ",speaker)

        for session in os.listdir(DIR_PATH + '/emgSync/' + speaker):
            print("  Processing session ",session,':')

            createFolders(session, speaker, 'features') 

            utt = 0
            numberOfUtterances = len(os.listdir(DIR_PATH + '/emgSync/' + speaker + '/' + session))
            printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)
            for file in os.listdir(DIR_PATH + '/emgSync/' + speaker + '/' + session):
                
                
                #print("     Processing: ",file)

                filename = DIR_PATH + '/emgSync/' + speaker + '/' + session + '/' + file

                signalSet = np.load(filename)
    
                labelSegments = getLabelSegments(filename,'emgSync','.npy')

                L = math.floor(frameSize*fs) # Length of frame in samples
                step = math.floor(frameShift*fs) # Length of frame shift in samples
                nSamples = np.shape(signalSet)[0] # Number of samples into the emg signals
                nFrames = int(math.ceil((nSamples-L)/step) + 1) # Number of frames resulting from the analysis with a window of choosen length and choosen frame shift
    
                nSignals = np.shape(signalSet[1])[0] - 1 # Number of emg signals in file. The last one (sync signal) is ignored
                TD0 = np.zeros((nSignals,nFrames,nFeatures),dtype='float32') # This matrix will contain the features of each frame of each signal

                # The features matrix will contain the features that are going to characterize each frame:
                # for each signal, the frame is passed by an stacking filter
                # Then, the stacked features of the frame for all signals are put together in the array corresponfig to the frame
                features = np.zeros((nFrames,nSignals*nFeatures*(2*k+1)+1),dtype='float32')
    
                # Features extraction for each frame in each signal
                for i in range(nSignals):
                    signalEMG = signalSet[:,i].astype(float)
                    signalEMG -= np.mean(signalEMG) # Remove DC
                    signalEMG = signalEMG/max(abs(signalEMG)) # Normalize
    
                    for j in range(nFrames):
                        start = j*step
                        end = start + L
    
                        # In the last frame shifting, if the window exceeds the signal
                        # the last sample of the signal is going to be repeated for filling the exceeding positions in frame
                        if end > len(signalEMG)-1:                            
                            xn = [signalEMG[-1]]*L
                            xn[0:len(signalEMG)-end] = signalEMG[start:]
    
                        else: # The normal case
                            xn = signalEMG[start:end]
    
                        # wn: double average of frame (used points are configured inside the function)
                        wn = calculateAverage(xn)
                        wn = calculateAverage(wn)
    
                        # pn = wn - xn
                        pn = np.subtract(xn,wn)
                        rn = np.abs(pn) # rn: rectified pn
        
                        featuresDict = {}

                        if 'Mw' in FEATURES_PER_FRAME:
                            featuresDict['Mw'] = sum(wn)/len(wn)
                        if 'Mr' in FEATURES_PER_FRAME:
                            featuresDict['Mr'] = sum(rn)/len(rn)
                        if 'Pw' in FEATURES_PER_FRAME:
                            featuresDict['Pw'] = sum( [ abs(x)**2 for x in wn ] ) / len(wn)
                        if 'Pr' in FEATURES_PER_FRAME:
                            featuresDict['Pr'] = sum( [ abs(x)**2 for x in rn ] ) / len(rn)
                        if 'zp' in FEATURES_PER_FRAME:
                            featuresDict['zp'] = zeroCrossingCount(pn)
    
                        for count, element in enumerate(FEATURES_PER_FRAME):
                            TD0[i,j,count] = featuresDict[element]
    
                # After features for all frames in all signals have been calculated, the 'features' matrix is going to be filled
                for j in range(nFrames):
                    phoneLabel = getPhoneLabel(j,L,step,fs,labelSegments)
                    features[j,0] = float(phoneDict[phoneLabel])

                    for i in range(nSignals):
                        start = i*nFeatures*(2*k+1)+1
                        end = (i+1)*nFeatures*(2*k+1)+1
                        features[j,start:end] = stackingFilter(TD0, i, j, k, nFeatures, nFrames)

                np.save(filename.replace('emgSync','features'),features)

                utt += 1
                printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)

def extractMFCCs():
    frameSize = AUDIO_FRAME_SIZE # In ms
    frameShift = AUDIO_FRAME_SHIFT # In ms
    k = STACKING_WIDTH # Width of stacking filter
    

    phoneDict = buildPhoneDict()

    for speaker in os.listdir(DIR_PATH + '/audioSync'):
        print("Processing speaker n. ",speaker)

        for session in os.listdir(DIR_PATH + '/audioSync/' + speaker):
            print("  Processing session ",session,':')

            createFolders(session, speaker, 'mfccs')

            utt = 0
            numberOfUtterances = len(os.listdir(DIR_PATH + '/emgSync/' + speaker + '/' + session))
            printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)
            for file in os.listdir(DIR_PATH + '/audioSync/' + speaker + '/' + session):

                if file.endswith(".wav"):

                    #print("     Processing: ",file)

                    filename = DIR_PATH + '/audioSync/' + speaker + '/' + session + '/' + file

                    fs, audioSignal = wavfile.read(filename) 
        
                    labelSegments = getLabelSegments(filename,'audioSync','.wav')

                    L = math.floor(frameSize*fs) # Length of frame in samples
                    step = math.floor(frameShift*fs) # Length of frame shift in samples
                    nSamples = np.shape(audioSignal)[0] # Number of samples into the emg signals
                    nFrames = int(math.ceil((nSamples-L)/step) + 1) # Number of frames resulting from the analysis with a window of choosen length and choosen frame shift
        
                    TD0 = np.zeros((1,nFrames,N_COEF),dtype='float32') # This matrix will contain the features of each frame of each signal

                    # The features matrix will contain the features that are going to characterize each frame:
                    # for each signal, the frame is passed by an stacking filter
                    # Then, the stacked features of the frame for all signals are put together in the array corresponfig to the frame
                    features = np.zeros((nFrames,N_COEF*(2*k+1)+1),dtype='float32')
        
                    TD0[0,:,:] = psf.mfcc(audioSignal, samplerate=fs, winlen=AUDIO_FRAME_SIZE, winstep=AUDIO_FRAME_SHIFT, numcep=N_COEF, nfilt=N_FILTERS, winfunc=np.hamming)
        
                    # After MFCC coefficients for all frames have been calculated, the 'features' matrix is going to be filled
                    for j in range(nFrames):
                        phoneLabel = getPhoneLabel(j,L,step,fs,labelSegments)
                        features[j,0] = float(phoneDict[phoneLabel])

                        features[j,1:] = stackingFilter(TD0, 0, j, k, N_COEF, nFrames)

                    np.save(filename.replace('audioSync','mfccs').replace('.wav','.npy'),features)

                    utt += 1
                    printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)

def main(extracted='features'):

    if len(sys.argv) > 1: # If function have been called from terminal, look into the given arguments
        option = sys.argv[1]
    else: # If function has been called from another script or not argument has been given, use the parameter instead 
        option = extracted

    if option == 'features': 
        extractFeatues()
    elif option == 'mfccs':        
        extractMFCCs()

if __name__ == '__main__':
    main()
