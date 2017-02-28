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
tmp = df[df.columns[1:]].corr()[attributesSet['Session Continues']]
print (tmp)

# numpyMatrix = df.as_matrix()
# [rowCount, columnCount] = numpyMatrix.shape
#
#
# # users without customer id are marked as '?'
#
#
# numCustId = len(set(numpyMatrix[:, attributesSet['Customer ID']]))
# sessionNumRegCust = len(numpyMatrix[numpyMatrix[:, attributesSet['Session Customer ID']] != '?'])
#
# """ basic informations"""
# numSessionCookieId = len(set(numpyMatrix[:,attributesSet['Session Cookie ID']]))
# numSessionId = len(set(numpyMatrix[:, attributesSet['Session ID']]))
# numSessionCustomerId = len(set(numpyMatrix[:, attributesSet['Session Customer ID']])) - 1
#
# tmpIndex = attributesSet['DoYouPurchaseForOthers']
# tmp = numpyMatrix[:, tmpIndex]
#
# print (numCustId, numSessionCookieId, numSessionId, numSessionCustomerId)
#
#
# """  finding of all records of specific customer"""
# # singleSessionRows = []
# # ids = numpyMatrix[:, attributesSet['Session Customer ID']]
# # for i in range (0, len(ids)):
# #     if ids[i] == "62":
# #         singleSessionRows.append(numpyMatrix[i])
# #         print(numpyMatrix[i][attributesSet['Session Continues']])
# #
# # print("aaa")
#
# sta = Stats(numpyMatrix, attributesSet)
# # sta.spearmanCorrelation()
# sta.test("WhichDoYouWearMostFrequent")






