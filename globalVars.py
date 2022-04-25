import math

DIR_PATH = '/mnt/ldisk/eder/EMG-UKA-Trial-Corpus' # The base path to the corpus to be processed
#DIR_PATH = '/mnt/ldisk/eder/ReSSInt/Pilot2/Session2'
SCRIPT_PATH = '/home/aholab/eder/scripts/EMG-UKA-Processing' # The path to the folder where those scripts are located

CORPUS = 'EMG-UKA' # Corpus being used. Expected values: 'EMG-UKA' or 'Pilot Study'

REMOVE_NUMBERS = True # When working with the EMG-UKA Trial Corpus, set this variable to True in order to remove the multiple numbered phonemes that the lexicon returns

MARK_CONTEXT_PHONEMES = False # When working with the Pilot Study Corpus, set this variable to True in order to mark the context phonemes
if MARK_CONTEXT_PHONEMES: # The MARK_TEXT_PHONEMES is not compatible with REMOVE_NUMBERS option, since marks are based on numbers. If not working with EMG-UKA Trial Corpus, this shouldn't be a problem
    REMOVE_NUMBERS = False

FS = 600 # Sampling frequency of the EMG signal
FRAME_SIZE = 0.025 # Size of the frame in ms
FRAME_SHIFT = 0.005 # Frame shift in ms
STACKING_WIDTH = 15 # Width of stacking filter
# 'backwards': Only the frames BEFORE the central frame are stacked
#       (final number of frames: STACKING_WIDTH + 1)
# 'symmetric': The same number of frames are stacked before and after the central frame
#       (final number of frames: 2*stacking filter width + 1)

STACKING_MODE = 'symmetric'

##########################################
#   PARAMETERS FOR HILBERT CALCULATION   #
##########################################

HILBERT_INTERMEDIATE_FS = 1000 # The Fs at which signal is resampled to calculate the Hilbert transform
HILBERT_END_FS = 200 # The Fs at which resulting Hilbert transform signal is resampled at the end


# Features:
# - Mw: Low Frequency Mean
# - Pw: Low Frequency Power
# - Mr: High Frequency Rectified Mean
# - Pr: High Frequency Rectified Power
# - zp: High Frequency Zero Crossing Rate

# In paper: ['Mw','Pw','Pr','zp','Mr']

FEATURE_NAMES = ['Mw','Pw','Pr','zp','Mr']

N_FEATURES = len(FEATURE_NAMES) # Number of calculated features

N_CHANNELS = 7 # Number of EMG channels (including syncronization channel)

#######################################
#   PARAMETERS FOR MFCC EXTRACTION    #
#######################################

AUDIO_FRAME_SIZE = 0.016
AUDIO_FRAME_SHIFT = 0.01
N_FILTERS = 30
N_COEF = 13