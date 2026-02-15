import numpy as np
from DataHandler import DataHandler

class Calculaitons(DataHandler): 
    def mean(self):
        return np.mean(self.data)
    def std(self):
        return np.std(self.data)
    def minData(self):
        return np.min(self.data)
    def maxData(self):
        return np.max(self.data)




