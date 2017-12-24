# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:56:41 2017

@author: RodierS
"""
import numpy as np
import Classifiers.Utilities as ut
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

trainingProportion = 0.9

trainingIndicesFile = "trainingIndices"
testIndicesFile = "testIndices"

tuned_parameters = [{'C':[1,10,100], 'penalty': ['l1','l2']}]
best_C=1 ##tuned on topics 4,5,6
best_penalty="l1" ##tuned on topics 4,5,6
best_params = 0

class TopicSVM:
    
    def __init__(self, afpList, embeddingsIndex, maxTokensInPost, numEmbeddingsDim, xfilename, yfilename, bTune, resultsFile):
        ut.writeToFile(resultsFile, "********** \n WORKING ON SVM TOPIC CLASSIFIER\n")
        self.xdata, self.ydata = ut.importData(afpList, maxTokensInPost, numEmbeddingsDim, embeddingsIndex, xfilename, yfilename, True)
        self.indices = getStartIndicesPerTopic(self.ydata)
        numPostsPerTopic = self.getNumPostsPerTopic(self.indices)
        self.setTrainingAndTestIndices(numPostsPerTopic)
        self.xTrain = ut.flatten(self.xdata[self.trainingIndices])
        self.yTrain = self.ydata[self.trainingIndices]
        self.xTest = ut.flatten(self.xdata[self.testIndices])
        self.yTest = self.ydata[self.testIndices]
        
        print(self.xTrain.shape)
        print(self.yTrain.shape)
        
        self.xdata = [] #clear up some memory
        self.ydata = [] #clear up some memory
        
        print("Fitting classifier...")
        if bTune == True:
            linearClassifier = self.tuneAndFitLinearSVC()
        else:
            linearClassifier = self.fitLinearSVC()

        y_true, y_pred = self.yTest, linearClassifier.predict(self.xTest)
        ut.writeToFile(resultsFile, classification_report(y_true, y_pred))
        ut.writeToFile(resultsFile, confusion_matrix(y_true, y_pred))
        
    
        
    def setTrainingAndTestIndices(self,numPostsPerTopic):
        self.trainingIndices = np.array([])
        self.testIndices = np.array([])
        for key, value in sorted(numPostsPerTopic.items()):
            startTrainIndex = self.indices[key]
            numPostsTrain = int(np.round_(value*trainingProportion,0))
            numPostsTest = int(value-numPostsTrain)
            endTrainIndex = int(startTrainIndex+numPostsTrain)
            startTestIndex = int(endTrainIndex)
            endTestIndex = int(startTestIndex+numPostsTest)
            self.trainingIndices = np.hstack((self.trainingIndices,
                       [x for x in range(startTrainIndex, endTrainIndex)]))
            self.testIndices = np.hstack((self.testIndices, 
                       [y for y in range(startTestIndex, endTestIndex)]))
        self.trainingIndices = self.trainingIndices.astype(int)
        self.testIndices = self.testIndices.astype(int)
        
                     
            
            
    def getNumPostsPerTopic(self,indices):
        print(indices)
        prevTopicId = 0
        prevValue = 0
        counter = 0
        numPosts = {}
        for key, value in sorted(indices.items()):
            if counter!=0:
                numPosts[prevTopicId] = value-prevValue
            if (counter+1)==len(indices):
                numPosts[key] = len(self.ydata)-value
            prevTopicId = key
            prevValue = value
            counter = counter+1
        return numPosts
    
    def tuneAndFitLinearSVC(self):
        clf = GridSearchCV(LinearSVC(dual=False), tuned_parameters, cv=3)
        clf.fit(self.xTrain, self.yTrain)
        bestParams = clf.best_params_
        print("Best params:")
        print(bestParams)
        print()
        return clf
        
    def fitLinearSVC(self):
        clf = LinearSVC(dual=False, C=best_C, penalty=best_penalty)
        clf.fit(self.xTrain, self.yTrain)
        return clf


def getStartIndicesPerTopic(ydata):
    indices = {}
    currentLabel = 0
    counter = 0
    for label in ydata:
        
        if label != currentLabel:
            currentLabel = label
            indices[label]=counter
        counter += 1
    return indices
    
        