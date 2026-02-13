import numpy as np
from DataHandler import DataHandler

class Histogram(DataHandler):
    def __init__(self, data, bins=None):
        self.data = data
        self.bins = bins
        self.range = (min(self.data), max(self.data))
        
    def get_histogram(self):
        bins = np.arange(600, 951, 50)

        hist, bin_edges = np.histogram(
            self.data,
            bins=bins,
            density=False
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 50

        return hist, bin_edges, (600, 950), bin_centers, bin_width

