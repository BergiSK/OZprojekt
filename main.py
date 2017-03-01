import pandas as pd
import numpy as np

attributeStartingLine = 8

booleanColumns = [4,12,24,25,26,31,32,33,34,35,36,37,38,39,47,51,58,60,61,62,63,64,65,66,67,80,127,236,271]
dateTimeColumns = [[88,89], [90,91], [144,145]]
categoryNotNanColumns = [103, 263, 289, 290, 291, 292, 293, 294, 295]
categoryWithNanColumns = [283, 284, 285]
idColumns = [23, 93, 94, 95]

# Loads column names of the dataset
def createAttributeSet():
    i = 0
    attributesSet = {}
    with open("dataset/question1agg.names", "r") as text:
        for line in text:
            if i >= attributeStartingLine:
                temp = line.split(':')
                # some attributes are suggested by dataset providers, to be ignored
                if "ignore" not in temp[1]:
                    attributesSet[line.split(':')[0]] = i - attributeStartingLine
            i += 1
    return attributesSet

def preprocessData(df):
    # replaces '?' with nans for numarical columns, nans with 'not_defined' for categorical columns,
    # nans with previous valid bool value for boolean columns
    df.replace('?', np.nan, inplace=True)
    df[categoryWithNanColumns].replace(np.nan, 'not_defined', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Convert object variables to numeric for correlation computation,
    # nans are replaced with mean value
    categoricVariableColumns = df.select_dtypes(['object']).columns
    df[categoricVariableColumns] = df[categoricVariableColumns].apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df.fillna(df.mean(), inplace=True)

    # Convert object variables to categoric for selected columns
    categoryNotNanColumns.extend(categoryWithNanColumns)
    for i in categoryNotNanColumns:
        df[i] = df[i].astype('category')

    # Remove useless data (continuous strings / ID variables)
    df.drop(df.columns[idColumns], axis=1, inplace=True)
    categoricVariableColumns = df.select_dtypes(['object']).columns
    df.drop(df.columns[categoricVariableColumns], axis=1, inplace=True)

    return df


def main():
    attributesSet = createAttributeSet()
    # Loads the data
    filename = "dataset/question1aggTimeStampParsed.csv"
    df = pd.read_csv(filename, sep=',',header=None, encoding = "ISO-8859-1", parse_dates=dateTimeColumns)

    df = preprocessData(df)

    # Compute correlation with predicted variable for all columns
    predictedVariableIndex = attributesSet['Session Continues']
    tmp = df.corr(method='spearman')[predictedVariableIndex]

    # get rid of correlation with itself (equals 1 anyway)
    print (tmp.drop(predictedVariableIndex).sort_values(ascending=False)[:20])
    print (tmp.shape)

    correlations = tmp.drop(predictedVariableIndex).sort_values(ascending=False)
    print(correlations)

    for item in correlations.index:
        print(list(attributesSet.keys())[list(attributesSet.values()).index(item)])

if __name__ == "__main__":
    main()







