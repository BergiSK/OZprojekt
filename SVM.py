import pandas as pd
import json
import numpy as np
from sklearn import svm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore
from scipy.stats import boxcox
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from configparser import ConfigParser
import codecs

parser = ConfigParser()
# Open the file with the correct encoding
with codecs.open('config.ini', 'r', encoding='utf-8') as f:
    parser.readfp(f)

dateTimeColumns = [0, 1, 2]
correlationThreshold = 0.15
pairwiseCorrelationThreshold = 0.5

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
    df = pd.read_csv(filename, sep=',', header=None, encoding="ISO-8859-1", parse_dates=dateTimeColumns)
    return df

def loadImportantColumns(df):
    columns = json.loads(parser.get("correlation","dict") )
    columns.sort(key=lambda x: abs(x[0]), reverse=True)
    numColumns = json.loads(parser.get("correlation", "numColumns"))

    used_columns = columns[:numColumns]
    # take just column names of df
    used_columns_numbers = [row[1] for row in used_columns]
    return df[used_columns_numbers], used_columns

def loadImportantColumnsForGeneration(df):
    columns = json.loads(parser.get("correlation","dict") )
    columns.sort(key=lambda x: abs(x[0]), reverse=True)
    numColumns = json.loads(parser.get("correlation", "numGeneratedColumns"))

    used_columns = columns[:numColumns]
    # take just column names of df
    used_columns = [row[1] for row in used_columns]
    return df[used_columns]

def generateteFeatures(matrix):
    poly = pp.PolynomialFeatures(2)
    res = poly.fit_transform(matrix)
    # df = pd.DataFrame(res)
    return res

def selectCorelationOnGeneratedFeatures(df, df_predicted_index):
    tmp = df.corr(method='spearman')[df.columns[df_predicted_index]]
    correlations = tmp.drop(df_predicted_index)
    bestCorrelations = []
    for i in range(correlations.size):
        if correlations[i] > correlationThreshold:
            bestCorrelations.append(np.array([correlations[i], i]))
    return np.array(bestCorrelations)


def deleteCorrelatedColumns(originalGeneratedDf, pairwiseCorrelation, correlationsToPredicted):
    # reduce pairwise correlation matrix to just triangle wiyhout diagonal
    reducedPairwiseCorrelations = (pairwiseCorrelation.where(np.triu(np.ones(pairwiseCorrelation.shape), k=1).astype(np.bool))
          .stack()
          .order(ascending=False))
    # row names contains two attribute names - the ones the correlation was calculated on
    rowNames = reducedPairwiseCorrelations.index.values
    for i in range(0, len(reducedPairwiseCorrelations)):
        # if pairwise correlation is bigger than threshold, delete one of attributes
        if reducedPairwiseCorrelations.values[i] > pairwiseCorrelationThreshold:
            correlation1 = correlationsToPredicted[rowNames[i][0]][1]
            correlation2 = correlationsToPredicted[rowNames[i][1]][1]
            # if correlation to prected variable is lower - delete this attribute
            if correlation1 > correlation2:
                # check if column exists
                if rowNames[i][1] in originalGeneratedDf.columns:
                    originalGeneratedDf.drop(rowNames[i][1], 1, inplace=True)
            else:
                if rowNames[i][0] in originalGeneratedDf.columns:
                    originalGeneratedDf.drop(rowNames[i][0], 1, inplace=True)
    return originalGeneratedDf


def createCorrelationListWithIndexes(originalColumnCorrelation, generatedColumnCorrelation):
    resultList = []
    for i in range(0, len(originalColumnCorrelation)):
        resultList.append([i, originalColumnCorrelation[i][0]])
    for i in range(0, generatedColumnCorrelation.shape[0]):
        resultList.append([len(originalColumnCorrelation) + i, generatedColumnCorrelation[i][0]])
    return resultList

def generateColumnsSelectBest(attributesSet, df):
    predictedVariableIndex = attributesSet['Session Continues']
    # rename predicted column to 'predicted'
    df.rename(columns={78: 'predicted'}, inplace=True)

    # predicted colum dataframe
    predictedColumnDf = df[:40000][df.columns[predictedVariableIndex]].as_matrix()
    # load orginal columns for generation
    test_df = loadImportantColumnsForGeneration(df)[:40000]
    # load orginal columns for classification + columns names with their correlation with predicted variable
    original_columns_df, bestCorrelationsOriginal = loadImportantColumns(df)[:40000]

    # get np.array of generated features
    generated_matrix = generateteFeatures(test_df.as_matrix())
    print("Number of generated columns: " + str(generated_matrix.shape[1]))
    # add predicted column
    test_matrix = np.c_[generated_matrix, predictedColumnDf]
    test_df = pd.DataFrame(test_matrix)

    # calculate correlations of generated columns with predicted columns
    bestCorrelationsGenerated = selectCorelationOnGeneratedFeatures(test_df, test_df.shape[1] - 1)

    important_generated_columns_df = test_df[bestCorrelationsGenerated[:, 1]]
    print("Number of generated after correlation with target variable: " + str(important_generated_columns_df.shape[1]))
    # original and generated values together
    originalGeneratedDf = pd.concat([original_columns_df, important_generated_columns_df], axis=1)
    # rename column names to sequence of numbers
    originalGeneratedDf.columns = range(0, originalGeneratedDf.columns.size)
    # pairwise correlation of all
    pairwiseCorrelation = originalGeneratedDf.corr(method='spearman').abs()
    # concat all correlations values to one list
    correlationsToPredicted = createCorrelationListWithIndexes(bestCorrelationsOriginal, bestCorrelationsGenerated)

    # delete pairwise correlated columns

    print("Number of all: " + str(originalGeneratedDf.shape[1]))
    return deleteCorrelatedColumns(originalGeneratedDf, pairwiseCorrelation, correlationsToPredicted)

# function responsible for returning accuracy, precision, recall and roc-auc score
def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred),
            precision_score(y, yPred, pos_label=0),
            recall_score(y, yPred, pos_label=0), roc_auc_score(y, yPred))

# scorer implementation for cross-validation, multiple scoring results
# need to be printed, since sklearn supports only one return value
def my_scorer(estimator, x, y):
    a, p, r, ra  = getScores(estimator, x, y)
    print (a, p, r, ra)
    return a+p+r+ra

def main():
    attributesSet = createAttributeSet()
    df = readDataFrame()[:40000]

    trainDf = generateColumnsSelectBest(attributesSet, df)
    print("Number of generated after pairwise correlation: " + str(trainDf.shape[1]))

    # normalize
    # numericColumns = trainDf.select_dtypes(['float64', 'int64'])
    # trainDf[trainDf.select_dtypes(['float64', 'int64']).columns] = (numericColumns - numericColumns.mean()) \
    #           / (numericColumns.max() - numericColumns.min())
    numeric_cols = trainDf.select_dtypes(['float64', 'int64']).columns
    trainDf[numeric_cols].apply(zscore)

    # shift data to positive for boxcox
    trainDf[numeric_cols] = trainDf[numeric_cols].add(2)

    # boxcox try
    for index in numeric_cols:
        if index != 185:
            trainDf[index] = boxcox(trainDf[index])[0]

    testDf = df['predicted']

    # clf = svm.SVC(kernel='rbf', class_weight='balanced')
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # scoresSSS = cross_val_score(clf, trainDf, testDf, cv=sss.split(trainDf, testDf))
    # scoresSSS = cross_val_score(clf, trainDf, testDf, cv=sss.split(trainDf, testDf), scoring=my_scorer)

    # print(str(scoresSSS))

    parameters = {'kernel': ('rbf', 'poly'), 'C': [1, 10],
                  'gamma': [0.01, 0.001, 0.0001], 'class_weight':['balanced']}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters, n_jobs=6, cv=2)

    clf.fit(trainDf, testDf)
    print(clf.best_params_)

    print(clf.grid_scores_)
    print(clf.n_splits_)
    print(clf.n_jobs)

if __name__ == "__main__":
    main()