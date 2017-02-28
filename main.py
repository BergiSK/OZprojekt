import numpy as np
import pandas as pd
from Stats import *

attributeStartingLine = 8
# loads columns of the dataset
i = 0
attributesSet = {}
with open("dataset/question1agg.names","r") as text:
    for line in text:
        if i >= attributeStartingLine:
            temp = line.split(':')
            # some attributes are suggested by dataset providers, to be ignored
            if "ignore" not in temp[1]:
                attributesSet[line.split(':')[0]] = i - attributeStartingLine
        i+=1

filename = "dataset/question1agg.csv"

df = pd.read_csv(filename, sep=',',header=None, encoding = "ISO-8859-1")
predictedVariableIndex = attributesSet['Session Continues']
tmp = df.corr()[predictedVariableIndex]

# get rid of correlation with itself (equals 1 anyway)
print (tmp.drop(predictedVariableIndex).sort_values(ascending=False)[:20])

# """ basic informations"""
# numSessionCookieId = len(set(numpyMatrix[:,attributesSet['Session Cookie ID']]))
# numSessionId = len(set(numpyMatrix[:, attributesSet['Session ID']]))
# numSessionCustomerId = len(set(numpyMatrix[:, attributesSet['Session Customer ID']])) - 1
#
#
# print (numCustId, numSessionCookieId, numSessionId, numSessionCustomerId)






