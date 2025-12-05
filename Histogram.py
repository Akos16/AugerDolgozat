import numpy as np
from DataHandler import DataHandler

class Histogram(DataHandler):
    def __init__(self, data, bins=None):
        self.data = data
        self.bins = bins
        self.range = (min(self.data), max(self.data))
        self.hist = None
        self.bin_edges = None
    def create(self):
        self.hist, self.bin_edges = np.histogram(self.data, bins = self.bins, range = self.range)
        return self.hist, self.bin_edges
    def plot(self, ax):
        ax.hist(
            self.data,
            bins=self.bins,
            range=self.range,
            histtype='step',
            color='black',
            linewidth=2,
            label='Data histogram'
        )
        ax.legend()
        return ax