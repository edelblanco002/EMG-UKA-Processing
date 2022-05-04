from globalVars import FS, FRAME_SIZE, N_CHANNELS
import keras
import math
import numpy as np
import pickle as pkl
import random
import tables
import tensorflow as tf
import time

L = math.floor(FS*FRAME_SIZE*(N_CHANNELS-1))
BATCH_SIZE = 32
TABLE_PATH = '/home/aholab/eder/ldisk/EMG-UKA-Trial-Corpus/allFrames/framesTable.h5'

class DataGenerator(keras.utils.all_utils.Sequence):
    def __init__(self, listIdx, batch_size=32, dim=90, shuffle=True, tablePath='/home/aholab/eder/ldisk/EMG-UKA-Trial-Corpus/allFrames/framesTable.h5'):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.tablePath = tablePath
        self.shuffle = shuffle
        self.listIdx = listIdx
        self.on_epoch_end()

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.listIdx)/self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Take the indexes of the 
        indexes = self.listIdx[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__data_generation(indexes)

        # X is returned 2 times since this function is expected to return an input list with its respective label list
        # An autoencoder needs the output to be the same as input
        return X, X

    def __data_generation(self, idxs):
        # Generates data containing batch_size samples
        # EMG signals have been already normalized before being divided into frames

        X = np.empty((self.batch_size, self.dim), dtype='float32')

        # Generate data
        tableFile = tables.open_file(self.tablePath)
        table = tableFile.root.data
        
        # Read data from h5 table
        # The two first columns are discarded, since they contain the utterance identifier and the label
        for i, idx in enumerate(idxs):
            X[i,] = table[idx,2:]

        tableFile.close()

        return X

    def on_epoch_end(self):
        # Update indexes after each epoch
        if self.shuffle == True:
            random.shuffle(self.listIdx)

def contractive_loss(y_true, y_pred):
    # Customized loss function
    # It is the weighted sum of the mse and the Frobinian norm of the Jacobian of the bottleneck with respect of the input layer

    lam = 1e-4

    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    W = model.get_layer('bottleneck').weights[0]  # N x N_hidden
    W = tf.transpose(W)  # N_hidden x N
    h = model.get_layer('bottleneck').output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * tf.reduce_sum(dh**2 * tf.reduce_sum(W**2, axis=1), axis=1)

    totalLoss = mse + contractive

    return totalLoss

def getIdxLists(validationSize, tablePath):
    # Gets the list of indexes for the train and the validation subset, choosen randomly
    # Validation size is the percentage over the total number of available frames
    tableFile = tables.open_file(tablePath)
    table = tableFile.root.data
    nFrames = len(table)
    tableFile.close()

    # Get a list with all the indexes (from 0 to len(table)-1) and shuffle them
    allIdx = list(range(nFrames))
    random.shuffle(allIdx)

    # Validation subset takes the indexed located in the fist part of the index matrix
    # (it doesn't work), since indexes have been shuffled
    # This is the last index + 1 belonging to the validation subset, and the first one of the training subset
    endValIdx = round(nFrames*validationSize)

    return allIdx[endValIdx:], allIdx[:endValIdx]



# define encoder
visible = tf.keras.layers.Input(shape=(L,), name='visible')

# encoder level 1
e = tf.keras.layers.Dense(64)(visible)
e = tf.keras.layers.BatchNormalization()(e)
e = tf.keras.layers.LeakyReLU()(e)

# encoder level 2
e = tf.keras.layers.Dense(32)(e)
e = tf.keras.layers.BatchNormalization()(e)
e = tf.keras.layers.LeakyReLU()(e)

# bottleneck
bottleneck = tf.keras.layers.Dense(16, name='bottleneck')(e)

# decoder level 1
d = tf.keras.layers.Dense(32)(bottleneck)
d = tf.keras.layers.BatchNormalization()(d)
d = tf.keras.layers.LeakyReLU()(d)

# decoder level 2
d = tf.keras.layers.Dense(64)(d)
d = tf.keras.layers.BatchNormalization()(d)
d = tf.keras.layers.LeakyReLU()(d)

output = tf.keras.layers.Dense(L)(d)

model = tf.keras.models.Model(visible,output)

# Add custom loss to the model and compile
model.add_loss(contractive_loss(visible, output))
model.compile(optimizer='adam',loss=None)

# Get list of the IDs for train and validation subsets
trainIdxList, valIdxList = getIdxLists(validationSize=0.33,tablePath=TABLE_PATH)

# Create data generators
trainingGenerator = DataGenerator(trainIdxList, batch_size=BATCH_SIZE)
validationGenerator = DataGenerator(valIdxList, batch_size=BATCH_SIZE)

# Create a checkpoint with the best model
model_checkpoint = keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)

history = model.fit(trainingGenerator,
        validation_data=validationGenerator,
        epochs=200, verbose=2, callbacks=[model_checkpoint])

# Extract the encoder from the best model
bestModel = keras.models.load_model("my_checkpoint.h5")

bestVisible = bestModel.get_layer('visible').output
bestBottleneck = bestModel.get_layer('bottleneck').output

encoder = tf.keras.models.Model(inputs=bestVisible,outputs=bestBottleneck)

# Save the encoder and the training history
t = time.time()
export_path_keras = "./encoder_{}.h5".format(int(t))
export_path_history = "./history_{}.pkl".format(int(t))
print(export_path_keras)
with open(export_path_history,'wb') as file:
    pkl.dump(history,file)
encoder.save(export_path_keras)