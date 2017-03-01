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

filename = "dataset/question1agg3.csv"

df = pd.read_csv(filename, sep=',',header=None, encoding = "ISO-8859-1", parse_dates=[88])
predictedVariableIndex = attributesSet['Session Continues']

df = pd.concat([
    df.select_dtypes([], ['object']),
    df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1)
# df = df.reindex_axis(specificColumn.columns, axis=1)

cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

tmp = df.corr(method="spearman")[predictedVariableIndex]
correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)

print("-----------------------------")
print(correlations)
print("-----------------------------")
print(correlations.shape)

for item in correlations.index:
    print(list(attributesSet.keys())[list(attributesSet.values()).index(item)])
# tmp = specificColumn.corr(method="spearman")[predictedVariableIndex]
# correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)
# print(correlations)
"""
# get rid of correlation with itself (equals 1 anyway)
correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)
print(correlations)

for item in correlations.index:
    print(list(attributesSet.keys())[list(attributesSet.values()).index(item)])
"""


