# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 13:37:58 2017

@author: RodierS
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import Classifiers.Utilities as ut

import numpy as np

trainingProportion = 0.9
validationProportion = 0.15
maxCat = 9
nFeatures =5000
nTokens = 100
nEmbeddingDim = 50

class TopicNeuralNetwork():
    
    def __init__(self, afpList, embeddingsIndex, xfilename, yfilename, sModelType, bTopic, resultsFile):
        ut.writeToFile(resultsFile, "********** \n WORKING ON NN TOPIC CLASSIFIER\n")
        self.xdata, self.ydata = ut.importData(afpList, nTokens, nEmbeddingDim, embeddingsIndex, xfilename, yfilename, bTopic)    
        
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
                self.xdata, self.ydata, test_size=1-trainingProportion)
        
        self.xTrain, self.xVal, self.yTrain, self.yVal = train_test_split(
                self.xTrain, self.yTrain, test_size=validationProportion)

        if(sModelType=="simple"):
            model = getSimpleNNModel()
        elif (sModelType=="lstm"):
            model = getLSTMModel()
        elif (sModelType=="cnn"):
            model = getCNNModel()
        elif (sModelType=="lstm_cnn"):
            model = getLSTM_CNNModel()
        
        model.fit(self.xTrain, to_categorical(self.yTrain), epochs=10, batch_size=10, verbose=1, validation_data=(self.xVal, to_categorical(self.yVal)))
        
        y_true, y_pred = self.yTest, model.predict(self.xTest)
        
        y_predLabel = ut.convertPredictionProbabilitiesMatrixToPredictionArray(y_pred)
        
        ut.writeToFile(resultsFile, classification_report(y_true, y_predLabel))
        ut.writeToFile(resultsFile, confusion_matrix(y_true, y_predLabel))




def getSimpleNNModel():
    model = Sequential()
    model.add(Dense(10, input_shape=(nTokens,nEmbeddingDim,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(maxCat+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getLSTMModel():
    model = Sequential()
    model.add(LSTM(100, input_shape=(nTokens,nEmbeddingDim,)))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(maxCat+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getCNNModel():
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(nTokens,nEmbeddingDim,)))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128,5,activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128,3,activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(maxCat+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getLSTM_CNNModel():
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(nTokens,nEmbeddingDim,)))
    model.add(MaxPooling1D(5))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(maxCat+1, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

        
        
