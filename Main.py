from Histogram import Histogram
from Calculations import Calculaitons
from DataHandler import DataHandler
from Distributions import Gumbel
import matplotlib.pyplot as plt

    
asd = DataHandler('./data.txt')
XmaxData = asd.getData()

histogram = Histogram(data=XmaxData, bins=15, range=(min(XmaxData), max(XmaxData)))
gumbel = Gumbel(XmaxData)

fig, ax = plt.subplots()
histogram.plot(ax)
gumbel.plot(ax)
plt.show()
