from bar import printProgressBar
from scipy.fftpack.pseudo_diffs import shift
from globalVars import DIR_PATH, N_CHANNELS, REMOVE_NUMBERS, SCRIPT_PATH
from globalVars import FS, FRAME_SHIFT, FRAME_SIZE, FEATURE_NAMES, N_FEATURES, STACKING_WIDTH, STACKING_MODE
from globalVars import AUDIO_FRAME_SIZE, AUDIO_FRAME_SHIFT, N_FILTERS, N_COEF
from globalVars import HILBERT_INTERMEDIATE_FS, HILBERT_END_FS
import math
import numpy as np
import os
import python_speech_features as psf
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, resample
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

def calculateNewLength(nSamples, actualFs, newFs):
    duration = nSamples/actualFs
    newLength = round(duration*newFs)
    return newLength

def createFolders(session, speaker, folderName):
    try:
        os.makedirs(DIR_PATH + '/' + folderName + '/' + speaker + '/' + session)
    except OSError as error:
        pass
        #print(error)

def filterSignal(signal, fs, order=4, lowcut=0, highcut=0, btype='low'):
    # This function filters the signal with a Butterworth pass filter

    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    if btype == 'band':
        b, a = butter(order, [low, high], btype='band')
    elif btype == 'low':
        b, a = butter(order, low, btype='low')

    """# The lenght of the filtered signal must be, at least, the padlen size, which is max(len(b),len(a))*3.
    # If it is not, add zero padding
    padlen = max(len(b), len(a))*3
    if len(signal) < padlen:
        paddingAdded = True
        padLen = math.ceil((padlen - len(signal))/2)
        signal = np.pad(signal,padlen,'edge')
    else:
        paddingAdded = False"""

    filteredSignal = filtfilt(b, a, signal)

    """# If padding was applied, remove it
    if paddingAdded:
        filteredSignal = filteredSignal[padlen:-padlen]"""

    return filteredSignal

def hilbertTransform(signal):

    np.save(f"{SCRIPT_PATH}/signal.npy",signal)

    # Resample signal
    resampledSignal = resampleSignal(signal, actualFs=FS, newFs=HILBERT_INTERMEDIATE_FS)
    np.save(f"{SCRIPT_PATH}/resampledSignal.npy",resampledSignal)

    # Get Hilbert analytical signal and take absolute value
    analyticSignal = hilbert(resampledSignal)
    np.save(f"{SCRIPT_PATH}/analyticSignal.npy",analyticSignal)
    analyticSignal = np.abs(analyticSignal)

    # Apply a low filter with cut freq=20 Hz.
    filteredSignal = filterSignal(analyticSignal, fs=HILBERT_INTERMEDIATE_FS, order=6, lowcut=20, btype='low')

    np.save(f"{SCRIPT_PATH}/filteredSignal.npy",filteredSignal)
    
    # Downsample signal after calculating
    downsampledSignal = resampleSignal(filteredSignal, actualFs=HILBERT_INTERMEDIATE_FS, newFs=HILBERT_END_FS)

    np.save(f"{SCRIPT_PATH}/downsampledSignal.npy",downsampledSignal)

    return downsampledSignal

def resampleSignal(signal, actualFs, newFs):
    newLength = calculateNewLength(len(signal), actualFs, newFs)

    resampledSignal = resample(signal,newLength)

    return resampledSignal

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

    if STACKING_MODE == 'symmetric':
        tdN = [0]*(2*k+1)*nFeatures
        frameRange = range(0,2*k + 1)
    elif STACKING_MODE == 'backwards':
        tdN = [0]*(k+1)*nFeatures
        frameRange = range(0,k + 1)

    for ind in frameRange:
        jj = j - k + ind
        
        if jj < 0 or jj >= nFrames: # If the range of the stacking filter exceeds the frame position,
            tdN[ind*nFeatures:(ind+1)*nFeatures] = [np.nan]*nFeatures # leave to 0 the features corresponding to those positions
        else:
            tdN[ind*nFeatures:(ind+1)*nFeatures] = TD0[i,jj,:]

    return tdN

def zeroCrossingCount(frame):
    # This function counts the zero crossings into the frame

    signs = np.sign(frame)
    signs[signs == 0] = -1
    return len(np.where(np.diff(signs))[0])

def extractFeatures():
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
                if STACKING_MODE == 'symmetric':
                    features = np.zeros((nFrames,nSignals*nFeatures*(2*k+1)+1),dtype='float32')
                elif STACKING_MODE == 'backwards':
                    features = np.zeros((nFrames,nSignals*nFeatures*(k+1)+1),dtype='float32')
    
                # Features extraction for each frame in each signal
                for i in range(nSignals):
                    signalEMG = signalSet[:,i].astype(float)
                    signalEMG -= np.mean(signalEMG) # Remove DC
                    signalEMG = signalEMG/max(abs(signalEMG)) # Normalize

                    # wn: double average of frame (used points are configured inside the function)
                    lowFrecSignal = calculateAverage(signalEMG)
                    lowFrecSignal = calculateAverage(lowFrecSignal)

                    # pn = wn - xn
                    highFrecSignal = np.subtract(signalEMG,lowFrecSignal)
                    # rn: rectified pn
                    rectHighFrecSignal = np.abs(highFrecSignal)

                    for j in range(nFrames):
                        start = j*step
                        end = start + L
    
                        # In the last frame shifting, if the window exceeds the signal
                        # the last sample of the signal is going to be repeated for filling the exceeding positions in frame
                        if end > len(lowFrecSignal)-1:
                            wn = [lowFrecSignal[-1]]*L
                            wn[0:len(lowFrecSignal)-end] = lowFrecSignal[start:]
                            pn = [highFrecSignal[-1]]*L
                            pn[0:len(highFrecSignal)-end] = highFrecSignal[start:]
                            rn = [rectHighFrecSignal[-1]]*L
                            rn[0:len(rectHighFrecSignal)-end] = rectHighFrecSignal[start:]
    
                        else: # The normal case
                            wn = lowFrecSignal[start:end]
                            pn = highFrecSignal[start:end]
                            rn = rectHighFrecSignal[start:end]
        
                        featuresDict = {}

                        if 'Mw' in FEATURE_NAMES:
                            featuresDict['Mw'] = sum(wn)/len(wn)
                        if 'Mr' in FEATURE_NAMES:
                            featuresDict['Mr'] = sum(rn)/len(rn)
                        if 'Pw' in FEATURE_NAMES:
                            featuresDict['Pw'] = sum( [ abs(x)**2 for x in wn ] ) / len(wn)
                        if 'Pr' in FEATURE_NAMES:
                            featuresDict['Pr'] = sum( [ abs(x)**2 for x in rn ] ) / len(rn)
                        if 'zp' in FEATURE_NAMES:
                            featuresDict['zp'] = zeroCrossingCount(pn)

                        for count, element in enumerate(FEATURE_NAMES):
                            TD0[i,j,count] = featuresDict[element]
                            count += 1

                # After features for all frames in all signals have been calculated, the 'features' matrix is going to be filled
                for j in range(nFrames):
                    phoneLabel = getPhoneLabel(j,L,step,fs,labelSegments)
                    features[j,0] = float(phoneDict[phoneLabel])

                    for i in range(nSignals):
                        if STACKING_MODE == 'symmetric':
                            start = i*nFeatures*(2*k+1)+1
                            end = (i+1)*nFeatures*(2*k+1)+1
                        elif STACKING_MODE == 'backwards':
                            start = i*nFeatures*(k+1)+1
                            end = (i+1)*nFeatures*(k+1)+1
                        features[j,start:end] = stackingFilter(TD0, i, j, k, nFeatures, nFrames)

                np.save(filename.replace('emgSync','features'),features)

                utt += 1
                printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)

def extractHilbertTransform():
    fs = HILBERT_END_FS
    frameSize = FRAME_SIZE
    frameShift = FRAME_SHIFT
    k = STACKING_WIDTH # Number of frames put together for each frame

    phoneDict = buildPhoneDict()

    for speaker in os.listdir(DIR_PATH + '/emgSync'):
        print("Processing speaker n. ",speaker)

        for session in os.listdir(DIR_PATH + '/emgSync/' + speaker):
            print("  Processing session ",session,':')

            createFolders(session, speaker, 'hilbert')

            utt = 0
            numberOfUtterances = len(os.listdir(DIR_PATH + '/emgSync/' + speaker + '/' + session))
            printProgressBar(utt, numberOfUtterances, prefix = '\tProgress:', suffix = f'{utt}/{numberOfUtterances}', length = 50)
            for file in os.listdir(DIR_PATH + '/emgSync/' + speaker + '/' + session):

                filename = DIR_PATH + '/emgSync/' + speaker + '/' + session + '/' + file

                signalSet = np.load(filename)
    
                labelSegments = getLabelSegments(filename,'emgSync','.npy')

                L = math.floor(frameSize*fs) # Length of frame in samples
                step = math.floor(frameShift*fs) # Length of frame shift in samples
                if step < 1:
                    step = 1
                nSamples = np.shape(signalSet)[0] # Number of samples into the emg signals
                nSamples = calculateNewLength(nSamples, actualFs=FS, newFs=fs)
                nFrames = int(math.ceil((nSamples-L)/step) + 1) # Number of frames resulting from the analysis with a window of choosen length and choosen frame shift

                nSignals = np.shape(signalSet[1])[0] - 1 # Number of emg signals in file. The last one (sync signal) is ignored

                # The features matrix will contain the segment of Hilbert transformed signal corresponding 
                # to (2k + 1) frames, with the actual frame in central position. The first column is for the label.
                features = np.zeros((nFrames,nSignals,L*(2*k+1)+1),dtype='float32')

                # Get Hilbert transform a segment for each frame in each signal
                for j in range(nFrames):
                    start = j*step - L*k # The window starts k frames before actual frame
                    end = start + L*(2*k + 1) # And ends (2k + 1) frames after the first frame

                    phoneLabel = getPhoneLabel(j,L,step,fs,labelSegments)

                    features[j,:,0] = float(phoneDict[phoneLabel])

                    for i in range(nSignals):
                        signalEMG = signalSet[:,i].astype(float)
                        signalEMG -= np.mean(signalEMG) # Remove DC
                        signalEMG = signalEMG/max(abs(signalEMG)) # Normalize
    
                        transformedSignal = hilbertTransform(signalEMG)

                        frame = np.zeros((L*(2*k + 1)))

                        # If the window lies out of signal, fill the resting array with NAN values
                        if start < 0:
                            frame[0:-start] = np.nan
                            frame[-start:] = transformedSignal[:end]
                        # In the last frame shifting, if the window exceeds the signal
                        # fill the window with nan values
                        elif end > len(transformedSignal):                            
                            frame[len(transformedSignal)-end:] = np.nan
                            frame[:len(transformedSignal)-end] = transformedSignal[start:]
    
                        else: # The normal case
                            frame = transformedSignal[start:end]

                        features[j,i,1:] = frame[:]

                np.save(filename.replace('emgSync','hilbert'),features)

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
                    if STACKING_MODE == 'symmetric':
                        features = np.zeros((nFrames,N_COEF*(2*k+1)+1),dtype='float32')
                    elif STACKING_MODE == 'backwards':
                        features = np.zeros((nFrames,N_COEF*(k+1)+1),dtype='float32')
        
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
        extractFeatures()
    elif option == 'mfccs':        
        extractMFCCs()
    elif option == 'hilbert':
        extractHilbertTransform()

if __name__ == '__main__':
    main()
