import matplotlib.pyplot as plt
from DataHandler import DataHandler
from Histogram import Histogram
from Distributions import Gumbel
from Plot import Plotter
data = DataHandler('./data.txt').getData()
bins = 15
data_range = (min(data), max(data))

fig, ax = plt.subplots()
plotter = Plotter(ax)

plotter.plot_histogram(data, bins, data_range)
plotter.plot_gumbel(Gumbel(data), bins, data_range)
plotter.finalize()

plt.show()