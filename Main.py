import matplotlib.pyplot as plt
from DataHandler import DataHandler
from Histogram import Histogram
from Distributions import Gumbel

asd = DataHandler('./data.txt')
data_array = asd.getData()  

fig, ax = plt.subplots()
Histogram(data_array, bins=15).plot(ax)
Gumbel(data_array).plot(ax)
plt.show()
