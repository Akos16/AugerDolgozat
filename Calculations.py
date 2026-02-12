import numpy as np
from DataHandler import DataHandler
import math
class Calculaitons(DataHandler): 
    def mean(self):
        return np.mean(self.data)
    def std(self):
        return np.std(self.data)
    def minData(self):
        return np.min(self.data)
    def maxData(self):
        return np.max(self.data)
    #def skewness(self):
        x1 = self.data
        n = len(x1)
        xMean = np.mean(x1)
        mtwo = mthree  = m2 = m3  = 0
        for i in x1:
            mtwo += math.pow(i - xMean, 2)
            mthree += math.pow(i - xMean, 3)
        m2 = mtwo / n
        m3 = mthree / n
        g1 = m3 / (m2 ** 1.5)
        G1 = math.sqrt(n * (n - 1)) / (n-2) * g1
        return G1
    #def kurtosis(self):
        x1 = self.data
        n = len(x1)
        xMean = np.mean(x1)
        mtwo = mfour  = m2 = m4  = 0
        for i in x1:
            mtwo += math.pow(i - xMean, 2)
            mfour += math.pow(i - xMean, 4)
        m2 = mtwo / n
        m4 = mfour / n
        kurtosis = m4 / (m2**2)
        excess_kurtosis = kurtosis - 3
        return excess_kurtosis




