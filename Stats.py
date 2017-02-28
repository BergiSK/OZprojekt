import numpy as np
import math
from scipy.stats.stats import pearsonr, spearmanr
class Stats:

    def __init__(self, matrix, attributesSet):
        self.matrix = matrix
        self.attributesSet = attributesSet

    def spearmanCorrelation(self):

        secondArray = self.matrix[:, self.attributesSet['Session Continues']]
        keys = list(self.attributesSet)

        for key in keys:
            if key == "Session Continues":
                continue

            firstArray = self.matrix[:, self.attributesSet[key]]
            if "?" in firstArray:
                firstArray[firstArray == "?"] = "not_defined"
            firstArray = self.replaceNaN(firstArray)

            res = spearmanr(firstArray, secondArray, nan_policy='omit')
            print("Column: " + key)
            print("correlation: " + str(res[0]) + "p-value: " + str(res[1]))
    def test(self, key):
        predictedVariable = self.matrix[:, self.attributesSet['Session Continues']]

        correlatedVariable = self.matrix[:, self.attributesSet[key]]
        if "?" in correlatedVariable:
            correlatedVariable[correlatedVariable == "?"] = "not_defined"

        firstArray = self.replaceNaN(correlatedVariable)
        res = spearmanr(firstArray, predictedVariable, nan_policy='omit')
        print("Column: " + key)
        print("correlation: " + str(res[0]) + " p-value: " + str(res[1]))




    def replaceNaN(self, array):
        i = 0
        value = 0
        if len(array[array == False]) > 0 or len(array[array == True]) > 0:
            value = True

        for i in range(0, len(array)):
            if str(array[i]) == 'nan':
                array[i] = value
        return array

