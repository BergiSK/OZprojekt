import pandas as pd
import numpy as np
import json
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import cross_validation

from configparser import ConfigParser
import codecs

parser = ConfigParser()
# Open the file with the correct encoding
with codecs.open('config.ini', 'r', encoding='utf-8') as f:
    parser.readfp(f)

dateTimeColumns = [0, 1, 2]

def createAttributeSet():
    i = 0
    attributesSet = {}
    with open("dataset/PreprocessedDataNames.txt", "r") as text:
        for line in text:
            attributesSet[line[:-1]] = i
            i += 1
    return attributesSet

def readDataFrame():
    # Loads the data
    filename = "dataset/PreprocessedData.csv"
    df = pd.read_csv(filename, sep=',', header=None, encoding="ISO-8859-1", parse_dates=dateTimeColumns,
                     dtype={54: 'category',
                            210: 'category',
                            212: 'category',
                            213: 'category',
                            214: 'category',
                            218: 'category',
                            219: 'category',
                            220: 'category',
                            221: 'category',
                            222: 'category',
                            223: 'category',
                            224: 'category'
                            })
    return df

def loadImportantColumns(df):
    columns = json.loads(parser.get("correlation","dict") )
    columns.sort(key=lambda x: abs(x[0]), reverse=True)
    numColumns = json.loads(parser.get("correlation", "numColumns"))

    used_columns = columns[:numColumns]
    # take just column names of df
    used_columns = [row[1] for row in used_columns]
    return df[used_columns]



def main():
    attributesSet = createAttributeSet()
    df = readDataFrame()

    trainDf = loadImportantColumns(df)
    testDf = df[78]

    clf = svm.SVC(kernel='rbf')
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
    scoresSSS = cross_validation.cross_val_score(clf, trainDf, testDf, cv=sss.split(trainDf, testDf))

    print(str(scoresSSS))

if __name__ == "__main__":
    main()