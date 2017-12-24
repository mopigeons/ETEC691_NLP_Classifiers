# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:47:22 2017

@author: RodierS
"""

#import data from json into python
##into python objects
import Utils.Utils as Utils
from Classifiers.TopicSVM import TopicSVM
from Classifiers.TopicDeepLearning import TopicNeuralNetwork
from Classifiers.StanceDeepLearning import StanceNeuralNetwork
import numpy as np

embeddingsDim = 50 #valid options: 50,100, 200, 300
numTokens = 100 #close to overall median (71), very close to median of other subsets
embeddingsFile = "./Glove/glove.6B."+str(embeddingsDim)+"d.txt"
topicIds = [3,7,8,9]
numCategories = len(topicIds)
topic_xdataFilename = "xData_Embedded"
topic_ydataFilename = "yData_topic"
stance_xdataFilename = "xData_stance_"
stance_ydataFilename = "yData_stance_"

topicSVM_resultsFile = "topicSVM_results_output.txt"
topicNN_resultsFile = "topicNN_results_output.txt"
stanceNN_resultsFile = "stanceNN_results_output.txt"


POST_COUNT_PER_TOPIC_CUTOFF = 17000

def main():
    jsonData = []
    for id in topicIds:
        jsonData.append(Utils.importFileToList("./Data/topic_"+str(id)+".json"))
    afpList = Utils.jsonListToAFPList(jsonData)
    
    afpList = balanceTopicCount(
            POST_COUNT_PER_TOPIC_CUTOFF, afpList, getPostsPerTopic(afpList))
    
    print("AFP list size: " + str(len(afpList)))
    
    embeddingsIndex = Utils.loadWordEmbeddings(embeddingsFile)
    
    for i in range(0,24):
        #valid options for the third argument are: simple, cnn, lstm, lstm_cnn
        topicNNMain(afpList, embeddingsIndex, "lstm_cnn")
    for i in range(0,24):
        stanceNNMain(afpList, embeddingsIndex)
    for i in range(0,24):
        topicSVMMain(afpList, embeddingsIndex)
    
def getPostsPerTopic(afpList):
    topicCount = {}
    for post in afpList:
        if post.getTopicId() in topicCount.keys():
            topicCount[post.getTopicId()] += 1
        else:
            topicCount[post.getTopicId()] = 1
    return topicCount
  
def topicSVMMain(afpList, embeddingsIndex):
    TopicSVM(afpList, embeddingsIndex, numTokens, embeddingsDim, topic_xdataFilename, topic_ydataFilename, False, topicSVM_resultsFile)

def topicNNMain(afpList, embeddingsIndex, sModelType):
    TopicNeuralNetwork(afpList, embeddingsIndex, topic_xdataFilename, topic_ydataFilename, sModelType, True, topicNN_resultsFile)
    
def stanceNNMain(afpList, embeddingsIndex):
    StanceNeuralNetwork(afpList, embeddingsIndex, stance_xdataFilename, stance_ydataFilename, stanceNN_resultsFile)

# Reduces size of AFP List. Only includes topics where topic count > minTopicCount,
# and only includes MinTopicCount posts per topic.
def balanceTopicCount(minTopicCount, afpList, topicCountIndex):
    newAfpList = []
    for topicId in topicIds:
        if topicCountIndex.get(topicId)>=minTopicCount:
            counter=0
            for afp in afpList:
                if counter >= minTopicCount:
                    break
                if afp.getTopicId() == topicId:
                    newAfpList.append(afp)
                    counter += 1
    return newAfpList

if __name__ == "__main__":
    main()