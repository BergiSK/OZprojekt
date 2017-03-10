import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

def calculateAndPrintCorrelations(df, attributesSet):
    # Compute correlation with predicted variable for all columns
    predictedVariableIndex = attributesSet['Session Continues']
    tmp = df.corr(method='spearman')[predictedVariableIndex]

    # get rid of correlation with itself (equals 1 anyway)
    # print (tmp.drop(predictedVariableIndex).sort_values(ascending=False)[:20])
    # print (tmp.shape)

    correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)
    # print(correlations)
    for item in correlations.index.values:
        print(str(item))
    print ("-----------")
    for item in correlations.values:
        print(str(item))
    print("-----------")
    for item in correlations.index:
        print(list(attributesSet.keys())[list(attributesSet.values()).index(item)])

def plotPairs(df, columnNames):
    # plt.figure(figsize=(8, 6), dpi=80)
    axes = pd.tools.plotting.scatter_matrix(df[columnNames],alpha=0.2, figsize=(12, 12))
   # plt.tight_layout()
    plt.savefig('scatter_matrix.png')

def plotPairsSeaborn(df, columnNames):
    sns_plot = sns.pairplot(df[columnNames], hue="Session Continues", diag_kind="kde")
    sns_plot.savefig('scatter_matrix_seaborn.png')

def plotBox(df, columnNames):
    # normalize data - except target variable
    columnNames.remove('Session Continues')
    newDf = df[columnNames].copy()
    df_norm = (newDf - newDf.mean()) / (newDf.max() - newDf.min())
    df_norm['Session Continues'] = df['Session Continues']

    df_norm.boxplot()
    plt.savefig('boxplot.png')

def plotBoxSeaborn(df, columnNames):
    # normalize data - except target variable
    columnNames.remove('Session Continues')
    newDf = df[columnNames].copy()
    df_norm = (newDf - newDf.mean()) / (newDf.max() - newDf.min())
    df_norm['Session Continues'] = df['Session Continues']

    ax = sns.boxplot(data=df_norm, orient="v", palette="Set3", whis=3)
    ax.get_figure().savefig('boxplot_seaborn.png')



def main():
    attributesSet = createAttributeSet()
    df = readDataFrame()


    # importantColumns = [56,216,131,96,attributesSet['Session Continues']]
    # columnNames = []
    # for i in importantColumns:
    #     columnNames.append(list(attributesSet.keys())[list(attributesSet.values()).index(i)])
    # df.rename(columns=dict(zip(importantColumns, columnNames)), inplace=True)

    # plotPairs(df, columnNames)
    # plotBox
    # plotBoxSeaborn(df, columnNames)
    # plotPairsSeaborn(df, columnNames)
    calculateAndPrintCorrelations(df, attributesSet)




if __name__ == "__main__":
    main()