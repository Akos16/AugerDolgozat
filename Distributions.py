import numpy as np
from scipy.stats import gumbel_r
from DataHandler import DataHandler
from Calculations import Calculaitons

class Gumbel():
    def __init__(self, data, bin_width):
        self.data = np.array(data)
        self.n = len(self.data)
        self.bin_width = bin_width
        self.calc = Calculaitons(data)
        self.gamma = 0.5772156649 

    def gumbel_default_params(self):
        sigma = max(self.calc.std(), 1e-6)
        beta = sigma * np.sqrt(6) / np.pi
        mu = self.calc.mean() - self.gamma * beta
        return mu, beta

    def model(self, x, mu, beta, a):
        pdf = gumbel_r.pdf(x, loc=mu, scale=beta)
        return pdf * a

    def skewness(self):
        return self.calc.skewness()

    def kurtosis(self):
        return self.calc.kurtosis()
