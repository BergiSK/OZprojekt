import pandas as pd
import numpy as np

dateTimeColumns = [0, 1, 2]


def createAttributeSet():
    i = 0
    attributesSet = {}
    with open("dataset/PreprocessedDataNames.txt", "r") as text:
        for line in text:
            attributesSet[line[:-1]] = i
            i += 1
    return attributesSet


def main():
    attributesSet = createAttributeSet()

    # Loads the data
    filename = "dataset/PreprocessedData.csv"
    df = pd.read_csv(filename, sep=',', header=None, encoding="ISO-8859-1", parse_dates=dateTimeColumns,
                     dtype={53: 'category',
                            209: 'category',
                            211: 'category',
                            212: 'category',
                            213: 'category',
                            217: 'category',
                            218: 'category',
                            219: 'category',
                            220: 'category',
                            221: 'category',
                            222: 'category',
                            223: 'category'
                            })
    # Compute correlation with predicted variable for all columns
    predictedVariableIndex = attributesSet['Session Continues']
    tmp = df.corr(method='spearman')[predictedVariableIndex]

    # get rid of correlation with itself (equals 1 anyway)
    # print (tmp.drop(predictedVariableIndex).sort_values(ascending=False)[:20])
    # print (tmp.shape)

    correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)
    print(correlations)

    for item in correlations.index:
        print(list(attributesSet.keys())[list(attributesSet.values()).index(item)])




if __name__ == "__main__":
    main()