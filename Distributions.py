import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from Calculations import Calculaitons
from DataHandler import DataHandler
from scipy.special import zeta

class Gumbel(DataHandler):
    def __init__(self, data):
        self.data = np.array(data)
        self.calc = Calculaitons(data)
        self.gamma = 0.5772

    def gumbel_default_params(self):
        sigma = max(self.calc.std(), 1e-3)
        beta = sigma * np.sqrt(6) / np.pi
        mu = self.calc.mean() - beta * self.gamma
        return mu, beta

    def get_gumbel_pdf(self, x, bins, data_len):
        mu, beta = self.gumbel_default_params()
        bin_width = (x[-1] - x[0]) / bins
        pdf = gumbel_r.pdf(x, loc=mu, scale=beta)
        return pdf * data_len * bin_width
    def skewness(self):
        return 12 * np.sqrt(6) * zeta(3) / np.pi**3
    def median(self):
        mu, beta = self.gumbel_default_params()
        return mu - beta * np.log(np.log(2))
    def mean_gumbel(self):
        mu, beta = self.gumbel_default_params()
        return mu + beta * self.gamma