import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from Calculations import Calculaitons

class Gumbel:
    def __init__(self, data=None):
        self.data = np.array(data)
        self.calc = Calculaitons(self.data)
        self.gamma = 0.5772 
        self.loc = None
        self.scale = None

    def default_params(self):
        sigma = max(self.calc.std(), 1e-3)
        beta = sigma * np.sqrt(6) / np.pi
        mu = self.calc.mean() - beta * self.gamma
        self.loc, self.scale = mu, beta
        return mu, beta

    def plot(self, ax, bins=15, range=None, color='red'):
        if range is None:
            range = (self.calc.minData(), self.calc.maxData())

        XmaxRange = np.linspace(range[0], range[1], 400)
        bin_width = (range[1] - range[0]) / bins

        mu, beta = self.default_params()
        gumbel_pdf = gumbel_r.pdf(XmaxRange, loc=mu, scale=beta)
        gumbel_scaled = gumbel_pdf * len(self.data) * bin_width

        ax.plot(XmaxRange, gumbel_scaled, color=color, linewidth=2, label='Gumbel')
        ax.legend()
        return ax
