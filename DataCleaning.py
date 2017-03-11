import pandas as pd
import numpy as np

attributeStartingLine = 8

booleanColumns = [4,12,24,25,26,31,32,33,34,35,36,37,38,39,43,47,51,55,56,58,60,61,62,63,64,65,66,67,80,95,127]
dateTimeColumns = [[88,89], [90,91], [144,145]]
categoryNotNanColumns = [103, 263, 289, 290, 291, 292, 293, 294, 295]
categoryWithNanColumns = [283, 284, 285]
idColumns = [23, 93, 94]

# Loads column names of the dataset
def createAttributeSet():
    i = 0
    attributesSet = {}
    with open("dataset/question1agg.names", "r") as text:
        for line in text:
            if i >= attributeStartingLine:
                temp = line.split(':')
                attributesSet[line.split(':')[0]] = i - attributeStartingLine
            i += 1
    #  setting special names for date columns
    for item in dateTimeColumns:
        key1 = list(attributesSet.keys())[list(attributesSet.values()).index(item[0])]
        key2 = list(attributesSet.keys())[list(attributesSet.values()).index(item[1])]

        attributesSet[key1 + " time"] = str(attributesSet[key1]) + "_" + str(attributesSet[key2])
        del attributesSet[key1]
        del attributesSet[key2]

    return attributesSet

def preprocessData(df):
    # create custom column SessionRegistered
    df[95].values[df[95].values == "?"] = False
    df[95].values[df[95].values != False] = True
    df[95] = df[95].astype('bool')

    # replaces '?' with nans for numarical columns, nans with 'not_defined' for categorical columns,
    # nans with previous valid bool value for boolean columns
    df.replace('?', np.nan, inplace=True)
    df[categoryWithNanColumns].replace(np.nan, 'not_defined', inplace=True)
    # applied only on booleans
    # df.fillna(method='bfill', inplace=True)
    for column in booleanColumns:
        df[column].fillna(df[column].mode()[0], inplace=True)

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
    df.drop(idColumns, axis=1, inplace=True)
    categoricVariableColumns = df.select_dtypes(['object']).columns
    df.drop(list(categoricVariableColumns.values), axis=1, inplace=True)



    # select the float columns
    df_float = df.select_dtypes(include=[np.float]).columns.values
    # select non-numeric columns
    df_num = df.select_dtypes(include=[np.int64]).columns.values
    num_columns = np.append(df_num, df_float)
    # set values higher then 95% quantil to 95% quantil
    for item in num_columns:
        high_quantiles = df[item].quantile(0.95)
        # some columns have almost all values 0 a than max quantil is 0
        if high_quantiles == 0:
            continue

        outliers_hight = (df[item] > high_quantiles)
        df[item].mask(outliers_hight, high_quantiles,  inplace=True)

    return df


def main():
    attributesSet = createAttributeSet()
    # Loads the data
    filename = "dataset/question1aggTimeStampParsed.csv"
    df = pd.read_csv(filename, sep=',',header=None, encoding = "ISO-8859-1", parse_dates=dateTimeColumns)

    df = preprocessData(df)

    preprocessedColumnNames = []
    for item in df.columns.values:
        preprocessedColumnNames.append(list(attributesSet.keys())[list(attributesSet.values()).index(item)])

    namesFile = open('dataset/PreprocessedDataNames.txt', 'w')
    for i in range(0, len(preprocessedColumnNames)):
        namesFile.write("%s\n" % preprocessedColumnNames[i] )

    df.to_csv("dataset/PreprocessedData.csv", sep=',', encoding="ISO-8859-1" , header=False, index=False)

if __name__=="__main__":
    main()







