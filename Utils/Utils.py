# -*- coding: utf-8 -*-
"""
@author: RodierS
"""

from Utils.AnnotatedForumPost import AnnotatedForumPost

import json
import numpy as np

def importFileToList(jsonFilename):
    data = []
    with open(jsonFilename, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
    

def jsonListToAFPList(jsonData):
    afpList = []
    for topic in jsonData:
        for jAfp in topic:
            authorId = jAfp['authorId']
            textTokens = jAfp['textTokens']
            topicId = jAfp['topicId']
            topicStanceId = jAfp['topicStanceId']
            afp = AnnotatedForumPost(authorId, textTokens, topicId, topicStanceId)
            afpList.append(afp)
    return afpList
        
    
def loadWordEmbeddings(embeddingsFile):
    embeddingsIndex = {}
    with open(embeddingsFile, encoding="utf-8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            embeddingsIndex[word] = embedding
    return embeddingsIndex
        
