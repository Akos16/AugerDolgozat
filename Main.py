from Histogram import Histogram
from DataHandler import XmaxData
import matplotlib.pyplot as plt
asd = Histogram(data=XmaxData, bins=7, range=(min(XmaxData), max(XmaxData)))

fig, ax = plt.subplots()
asd.plot(ax)
plt.show()