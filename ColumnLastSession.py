import numpy as np
import pandas as pd


i = 0
attributesSet = {}
with open("dataset/question1agg.names","r") as text:
    for line in text:
        if i > 7:
            attributesSet[line.split(':')[0]] = i - 8
        i+=1

filename = "dataset/newDataset.csv"

df = pd.read_csv(filename, sep=',',header=None, encoding = "ISO-8859-1")
numpyMatrix = df.as_matrix()
[rowCount, columnCount] = numpyMatrix.shape

sessionCookieIDs = []

attributesSet["LastSession"] = len(attributesSet)
lastSession = np.empty((0), dtype=bool)
for i in range(rowCount - 1, -1, -1):
    if numpyMatrix[i][attributesSet['Session Cookie ID']] not in sessionCookieIDs:
        sessionCookieIDs.append(numpyMatrix[i][attributesSet['Session Cookie ID']])
        lastSession = np.append(lastSession, True)
        # print(numpyMatrix[i][attributesSet['Session Cookie ID']])

    else:
        lastSession = np.append(lastSession, False)
        # print(numpyMatrix[i][attributesSet['Session Cookie ID']])

reversedlastSession = lastSession[::-1]
numpyMatrix = np.append(numpyMatrix, np.array(lastSession, copy=False, subok=True, ndmin=2).T , axis=1)

np.savetxt('dataset/newDataset.csv', numpyMatrix, delimiter=',', fmt="%s")