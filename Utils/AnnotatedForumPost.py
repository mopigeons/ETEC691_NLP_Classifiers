# -*- coding: utf-8 -*-
"""

@author: RodierS
"""
#%%
import numpy as np


class AnnotatedForumPost:
    LRB = "-LRB-"
    RRB = "-RRB-"
    LSB = "-LSB-"
    RSB = "-RSB-"
    
    
    def __init__(self, authorId, textTokens, topicId, topicStanceId):
        self._authorId = authorId
        self.setTextTokens(textTokens)
        self._topicId = topicId
        self._topicStanceId = topicStanceId
        self._embeddings = []
    
    def toString(self):
        st = str(self._authorId) + str(self._textTokens) + str(self._topicId) + str(self._topicStanceId)
        return st
    
    def getAuthorId(self):
        return self._authorId
    
    def getTextTokens(self):
        return self._textTokens

    def setTextTokens(self, textTokens):
        tokens = []
        for token in textTokens:
            if(token=="-LRB-"):
                tokens.append("(")
            elif(token=="-RRB-"):
                tokens.append(")")
            elif(token=="-LSB-"):
                tokens.append("[")
            elif(token=="-RSB-"):
                tokens.append("]")
            else:
                tokens.append(token)
        self._textTokens = tokens
    
    def getTopicId(self):
        return self._topicId
    
    def getTopicStanceId(self):
        return self._topicStanceId
    
    def getNumberOfTokens(self):
        return len(self._textTokens)
   