import numpy as np
from Calculations import Calculaitons

#Test data for skew and kurt
np.random.seed(42)
n = 100000
mu = 0
sigma = 1
data = np.random.normal(mu, sigma, n)

calc = Calculaitons(data)

skew = calc.skewness()
kurt = calc.kurtosis()
print(f'Skewness: {skew}, Kurtosis: {kurt}') #Megfelelő adatokat ad vissza
