import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from DataHandler import DataHandler
from Histogram import Histogram
from Distributions import Gumbel
from main import Ebins

Ebins = Ebins
bins = 20


fig, axes = plt.subplots(2, 4)
axes = axes.flatten()

mng = plt.get_current_fig_manager()
screen_width, screen_height = mng.window.winfo_screenwidth(), mng.window.winfo_screenheight()

fig_width = screen_width * 0.8 / mng.canvas.figure.dpi  # inch-be átváltva
fig_height = screen_height * 0.8 / mng.canvas.figure.dpi
fig.set_size_inches(fig_width, fig_height)

for i in range(8): 
    
    filename = f"./XmaxDists/XmaxDist_Ebin{i}.txt"
    Xmax, Counts, CountsSqrt = DataHandler(filename).getData()
    data = np.repeat(Xmax, Counts.astype(int))
    print(data)


    hist_obj = Histogram(data, bins=bins)

    hist, bin_edges, data_range, bin_centers, bin_width = hist_obj.get_histogram()
    #widths = np.diff(bin_edges)
    #hist = hist / (np.sum(hist) * bin_width)
    gumbObj = Gumbel(data, bin_width)
    mu, beta = gumbObj.gumbel_default_params()
    x_data=bin_centers
    y_data=hist
    a = np.max(hist)

    sigma = np.sqrt(y_data)
    sigma[sigma == 0] = 1.0
    popt, pcov = curve_fit(gumbObj.model, x_data, y_data, p0=[mu, beta, a], sigma=sigma, absolute_sigma=True)

    print(popt)
    print(pcov)

    skew = gumbObj.skewness()
    kurt = gumbObj.kurtosis()

    perr = np.sqrt(np.diag(pcov))

    print(f"Optimal parameters: mu = {popt[0]}, beta = {popt[1]}, amplitudo = {popt[2]}")
    print(f"Standard errors: mu_err = {perr[0]}, beta_err = {perr[1]}, amplitudo_err = {perr[2]}")

    mu, beta, a = popt
    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = gumbObj.model(x_model, mu, beta, a)

    dy2 = perr[0] + perr[1] + pcov[0][1] + perr[2] + pcov[1][2] + pcov[0][2];
    dy = np.sqrt(dy2);
    print(dy)
    ax = axes[i]
    ax.plot(x_model, y_model, color='red', linewidth=1, label='Gumbel', zorder=5)
    ax.fill_between(x_model, y_model-dy, y_model+dy)
    ax.hist(data, bins, histtype='step', color='black', linewidth=2)
    ax.text(
    0.60, 0.90,
    f'Skewness = {skew:.2f}\nKurtosis = {kurt:.2f}',
    transform=ax.transAxes,
    verticalalignment='top'
)
    E_low = Ebins[i]
    E_high = Ebins[i+1]
    ax.set_title(f'Energy: {E_low:.2f} – {E_high:.2f} lg(E/eV)')
    ax.legend()

plt.tight_layout()
plt.show()
