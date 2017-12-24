# -*- coding: utf-8 -*-
"""

@author: RodierS
"""

import numpy as np

def writeToFile(resultsFile,contents):
    f = open(resultsFile, 'a+')
    if isinstance(contents, np.ndarray):
        opt = np.get_printoptions()
        np.set_printoptions(threshold=np.inf)
        f.write(np.array2string(contents))
        f.write("\n")
        np.set_printoptions(**opt)
    else:
        f.write(contents+"\n")
    f.close()
    
def importData(afpList, nTokens, nEmbeddingDim, embeddingsIndex, xfilename, yfilename, bTopic):
    if(bTopic == True):
        xdata = np.zeros((len(afpList),nTokens,nEmbeddingDim))
        ydata = np.zeros(len(afpList))
        aCounter=0
        for afp in afpList:
            xItem = np.zeros((nTokens, nEmbeddingDim))
            tCounter = 0
            for token in afp.getTextTokens():
                tokenLowercase = token.lower()
                if tCounter >= nTokens:
                    break
                if tokenLowercase in embeddingsIndex:
                    xItem[tCounter] = embeddingsIndex[tokenLowercase]
                tCounter = tCounter + 1
            ydata[aCounter] = afp.getTopicId()
                
            xdata[aCounter] = xItem
            aCounter = aCounter+1
        return xdata, ydata
    else: ## stance
        afpByTopic = {}
        for afp in afpList:
            topicId = afp.getTopicId()
            if topicId not in afpByTopic:
                afpByTopic[topicId] = []
            topicStanceId = afp.getTopicStanceId()
            if topicStanceId != 0 and topicStanceId!=9999 and topicStanceId is not None:
                afpByTopic[topicId].append(afp)
        xdata = {}
        ydata = {}
        for topicId, afpList in afpByTopic.items():
            
            if topicId not in xdata:
                xdata[topicId] = np.zeros((len(afpList), nTokens, nEmbeddingDim))
                ydata[topicId] = np.zeros(len(afpList))
                aCounter=0
                for afp in afpList:
                    xItem = np.zeros((nTokens, nEmbeddingDim))
                    tCounter=0
                    for token in afp.getTextTokens():
                        tokenLowercase = token.lower()
                        if tCounter >= nTokens:
                            break
                        if tokenLowercase in embeddingsIndex:
                            xItem[tCounter] = embeddingsIndex[tokenLowercase]
                        tCounter= tCounter+1
                    topicStanceId = afp.getTopicStanceId()
                    
                    
                    ydata[topicId][aCounter] = afp.getTopicStanceId()
                    xdata[topicId][aCounter] = xItem
                    aCounter = aCounter + 1
        return xdata, ydata
        
            
                
                
            

def convertPredictionProbabilitiesMatrixToPredictionArray(y_pred):
    y_output = np.zeros(y_pred.shape[0])
    row_ind = 0
    for row in y_pred:
        maxPred = 0
        predInd = 0
        currentInd = 0
        for pred in row:
            if pred>maxPred:
                maxPred = pred
                predInd = currentInd
            currentInd = currentInd + 1
        y_output[row_ind] = predInd
        row_ind = row_ind + 1
    return y_output
    
def flatten(data):
    flattened_data = np.zeros((data.shape[0], data.shape[1]*data.shape[2]))
    i = 0
    for instance in data:
        flattened_data[i] = instance.flatten()
        i = i+1
    return flattened_data
    

    