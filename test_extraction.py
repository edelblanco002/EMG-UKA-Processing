import extractFeatures
import numpy as np

def test_calculateAverage():
    
    sig = np.random.uniform(low=-1,high=1,size=(20,))
    
    trueResult = [0]*20

    trueResult[0] = (sig[0] + sig[1] + sig[2] + sig[3] + sig[4])/9
    trueResult[1] = (sig[0] + sig[1] + sig[2] + sig[3] + sig[4] + sig[5])/9
    trueResult[2] = (sig[0] + sig[1] + sig[2] + sig[3] + sig[4] + sig[5] + sig[6])/9
    trueResult[3] = (sig[0] + sig[1] + sig[2] + sig[3] + sig[4] + sig[5] + sig[6] + sig[7])/9
    trueResult[4] = (sig[0] + sig[1] + sig[2] + sig[3] + sig[4] + sig[5] + sig[6] + sig[7] + sig[8])/9
    trueResult[5] = (sig[1] + sig[2] + sig[3] + sig[4] + sig[5] + sig[6] + sig[7] + sig[8] + sig[9])/9
    trueResult[6] = (sig[2] + sig[3] + sig[4] + sig[5] + sig[6] + sig[7] + sig[8] + sig[9] + sig[10])/9
    trueResult[7] = (sig[3] + sig[4] + sig[5] + sig[6] + sig[7] + sig[8] + sig[9] + sig[10] + sig[11])/9
    trueResult[8] = (sig[4] + sig[5] + sig[6] + sig[7] + sig[8] + sig[9] + sig[10] + sig[11] + sig[12])/9
    trueResult[9] = (sig[5] + sig[6] + sig[7] + sig[8] + sig[9] + sig[10] + sig[11] + sig[12] + sig[13])/9
    trueResult[10] = (sig[6] + sig[7] + sig[8] + sig[9] + sig[10] + sig[11] + sig[12] + sig[13] + sig[14])/9
    trueResult[11] = (sig[7] + sig[8] + sig[9] + sig[10] + sig[11] + sig[12] + sig[13] + sig[14] + sig[15])/9
    trueResult[12] = (sig[8] + sig[9] + sig[10] + sig[11] + sig[12] + sig[13] + sig[14] + sig[15] + sig[16])/9
    trueResult[13] = (sig[9] + sig[10] + sig[11] + sig[12] + sig[13] + sig[14] + sig[15] + sig[16] + sig[17])/9
    trueResult[14] = (sig[10] + sig[11] + sig[12] + sig[13] + sig[14] + sig[15] + sig[16] + sig[17] + sig[18])/9
    trueResult[15] = (sig[11] + sig[12] + sig[13] + sig[14] + sig[15] + sig[16] + sig[17] + sig[18] + sig[19])/9
    trueResult[16] = (sig[12] + sig[13] + sig[14] + sig[15] + sig[16] + sig[17] + sig[18] + sig[19])/9
    trueResult[17] = (sig[13] + sig[14] + sig[15] + sig[16] + sig[17] + sig[18] + sig[19])/9
    trueResult[18] = (sig[14] + sig[15] + sig[16] + sig[17] + sig[18] + sig[19])/9
    trueResult[19] = (sig[15] + sig[16] + sig[17] + sig[18] + sig[19])/9

    result = extractFeatures.calculateAverage(sig)

    assert trueResult == result

def test_getLabelsSegment():
    filename = 'phones_test.lab'

    trueResult = [[0.1, 'sil'], [0.2, 'H'], [0.3, 'OH'], [0.35, 'L'], [0.4, 'AH'], [0.5, 'sp']]

    result = extractFeatures.getLabelSegments(filename)

    assert trueResult == result

def test_getPhoneLabel():
    labelSegments = [[0.1, 'sil'], [0.2, 'H'], [0.3, 'OH'], [0.35, 'L'], [0.4, 'AH'], [0.5, 'sp']]

    L = 0.05
    step = 0.025
    fs = 1

    trueResult = ['sil', 'sil', 'sil', 'sil+H', 'sil-H', 'H', 'H', 'H+OH', 'H-OH', 'OH', 'OH', 'OH-L', 'L+AH', 'L-AH', 'AH', 'AH+sp', 'AH-sp', 'sp', 'sp']

    result = []

    for j in range(0,19):
        result.append(extractFeatures.getPhoneLabel(j,L, step, fs, labelSegments))

    assert trueResult == result

def test_zeroCrossingCount():
    time = np.arange(0,10,0.1)

    amplitude = np.sin(time)

    result = extractFeatures.zeroCrossingCount(amplitude)

    trueResult = 4

    assert result == trueResult