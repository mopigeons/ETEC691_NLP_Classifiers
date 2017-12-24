# -*- coding: utf-8 -*-
"""
@author: RodierS
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.utils import class_weight

import Classifiers.Utilities as ut

import numpy as np


trainingProportion = 0.9
validationProportion = 0.15
nFeatures =5000
nTokens = 100
nCat = 3
nTopic = 4
nEmbeddingDim = 50


class StanceNeuralNetwork:
    
    def __init__(self, afpList, embeddingsIndex, xfilename, yfilename, resultsFile):
        ut.writeToFile(resultsFile, "************ \n WORKING ON STANCE NEURAL NETWORK")
        self.xdata, self.ydata = ut.importData(afpList, nTokens, nEmbeddingDim, embeddingsIndex, xfilename, yfilename, False)
        total = 0
        for key, value in self.xdata.items():
            xTrain, xTest, yTrain, yTest = train_test_split(
                self.xdata[key], self.ydata[key], test_size=1-trainingProportion)
        
            xTrain, xVal, yTrain, yVal = train_test_split(
                    xTrain, yTrain, test_size=validationProportion)
            
            model = getLSTM_CNNModel()
            
            labels = np.unique(yTrain)
            
            
            classweight = class_weight.compute_class_weight('balanced', np.unique(yTrain), yTrain)
#            classweight = dict(enumerate(classweight))
            
            relabeled_cw = {}
            counter=0
            for label in labels:
                relabeled_cw[label]=classweight[counter]
                counter +=1
            classweight = relabeled_cw
            
            model.fit(xTrain, to_categorical(yTrain), epochs=20, batch_size=10, verbose=1, validation_data=(xVal, to_categorical(yVal)), class_weight=classweight)
            
            y_true, y_pred = yTest, model.predict(xTest)
        
            y_predLabel = ut.convertPredictionProbabilitiesMatrixToPredictionArray(y_pred)
            
            ut.writeToFile(resultsFile, "*****\n RESULTS FOR TOPIC "+str(key))
            ut.writeToFile(resultsFile, classification_report(y_true, y_predLabel))
            ut.writeToFile(resultsFile, confusion_matrix(y_true, y_predLabel))
        
        
        
    def removeInstancesWithoutLabel(self):
        indicesToDelete = []
        for i in range(0, self.xdata.shape[0]):
            if np.isnan(self.ydata[i]) or self.ydata[i]==0:
                indicesToDelete.append(i)
        print(len(indicesToDelete))
        self.xdata = np.delete(self.xdata, indicesToDelete, axis=0).astype(int)
        self.ydata = np.delete(self.ydata, indicesToDelete, axis=0).astype(int)
            
        
def getSimpleNNModel():
    model = Sequential()
    model.add(Dense(10, input_shape=(nTokens, nEmbeddingDim,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(nCat+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model       
        


def getLSTM_CNNModel():
    model = Sequential()
    model.add(Conv1D(256, 5, activation='relu', input_shape=(nTokens,nEmbeddingDim,)))
    model.add(MaxPooling1D(5))
    model.add(LSTM(200))
    model.add(Dropout(0.25))
    model.add(Dense(nCat+1, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model