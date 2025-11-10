import numpy as np
class Calculaitons: 
    def __init__(self, data=None):
        self.data = np.array(data)
    def mean(self):
        return np.mean(self.data)
    def std(self):
        return np.std(self.data)
    def minData(self):
        return np.min(self.data)
    def maxData(self):
        return np.max(self.data)
        