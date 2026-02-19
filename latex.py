import numpy as np
import matplotlib.pyplot as plt
'''
mu, beta = 0, 0.1 # location and scale
s = np.random.gumbel(mu, beta, 1000)

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
         * np.exp( -np.exp( -(bins - mu) /beta) ),
         linewidth=2, color='r')
plt.show()

'''
scores = [45, 50, 62, 70, 72, 75, 80, 85, 90, 95]

plt.figure()
plt.hist(scores, bins=5)

plt.title("Dolgozatpontszámok hisztogramja")
plt.xlabel("Pontszám")
plt.ylabel("Gyakoriság")

plt.show()


