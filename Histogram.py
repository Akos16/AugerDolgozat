import numpy as np

#fit
class Histogram:
    def __init__(self, data=None, bins=None, range=None):
        self.data = np.array(data)
        self.bins = bins
        self.range = range
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