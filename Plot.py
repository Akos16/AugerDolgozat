import numpy as np
from Distributions import Gumbel
class Plotter:
    
    def __init__(self, ax):
        self.ax = ax

    def plot_histogram(self, data, bins, data_range):
        skew = Gumbel(data).skewness()
        median = Gumbel(data).median()
        mean = np.mean(data)

        label = (
            r'Data histogram'
            f'\nSkewness = {skew:.5f}'
            f'\nMedian = {median:.5f}'
            f'\nMean = {mean:.5f}'
        )

        self.ax.hist(
            data,
            bins=bins,
            range=data_range,
            histtype='step',
            color='black',
            linewidth=2,
            label=label
        )
        self.ax.legend(
            loc='upper right',
            frameon=True,
            fontsize=9,           
            labelspacing=0.5       
        )
    def plot_gumbel(self, gumbel_obj, bins, data_range):
        x = np.linspace(data_range[0], data_range[1], 400)
        y = gumbel_obj.get_gumbel_pdf(x, bins, len(gumbel_obj.data))

        self.ax.plot(x, y, color='red', linewidth=2, label='Gumbel')

    def finalize(self):
        self.ax.legend()
