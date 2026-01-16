import numpy as np
from DataHandler import DataHandler

class Histogram(DataHandler):
    def __init__(self, data, bins=None):
        self.data = data
        self.bins = bins
        self.range = (min(self.data), max(self.data))

    def get_histogram(self):
        hist, bin_edges = np.histogram(
            self.data,
            bins=self.bins,
            range=self.range
        )
        return hist, bin_edges, self.range
