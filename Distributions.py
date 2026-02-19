import numpy as np
from scipy.stats import gumbel_r

class Gumbel():
    def __init__(self):
        self.gamma = 0.5772156649
    def model(self, x, mu, beta, a):
        pdf = gumbel_r.pdf(x, loc=mu, scale=beta)
        return pdf * a    
    def gumbel_pdf(self, x, mu, beta):
        z = (x - mu) / beta
        return (1.0 / beta) * np.exp(-(z + np.exp(-z)))
    def skewness_fit(self, mu, beta):
        x = np.linspace(mu - 20*beta, mu + 20*beta, 20000)
        pdf = self.gumbel_pdf(x, mu, beta)

        mean = np.trapz(x * pdf, x)
        var = np.trapz((x - mean)**2 * pdf, x)
        std = np.sqrt(var)

        mu3 = np.trapz((x - mean)**3 * pdf, x)
        return mu3 / std**3, mean, var
    def kurtosis_fit(self, mu, beta):
        x = np.linspace(mu - 20*beta, mu + 20*beta, 20000)
        pdf = self.gumbel_pdf(x, mu, beta)

        mean = np.trapz(x * pdf, x)
        var = np.trapz((x - mean)**2 * pdf, x)
        std = np.sqrt(var)

        mu4 = np.trapz((x - mean)**4 * pdf, x)
        return mu4 / std**4 - 3
