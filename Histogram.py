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
            range=self.range,
            density=False
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        return hist, bin_edges, self.range, bin_centers, bin_width
