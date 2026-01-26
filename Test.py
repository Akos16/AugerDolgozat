import numpy as np
import math
from scipy.stats import skew
from scipy.stats import gumbel_r

#gumbel adatok
x1 = gumbel_r.rvs(loc=800, scale=40, size=1000000)
n = len(x1)
xMean = np.mean(x1);
#szoras, skewness, kurtosis
mtwo = mthree = mfour = m2 = m3 = m4 = 0
m4 = 0
for i in x1:
    mtwo += math.pow(i - xMean, 2)
    mthree += math.pow(i - xMean, 3)
    mfour += math.pow(i - xMean, 4)
m2 = mtwo / n
m3 = mthree / n
m4 = mfour / n

#Fisher-Pearson coefficient of skewness
g1 = m3 / (m2 ** 1.5)
print("Fisher-Pearson coefficient of skewness: ", g1)
#adjusted Fisher-Pearson coefficient of skewness
G1 = math.sqrt(n * (n - 1)) / (n-2) * g1
print("adjusted Fisher-Pearson coefficient of skewness", G1)
