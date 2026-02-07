import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from DataHandler import DataHandler
from Histogram import Histogram
from Distributions import Gumbel
from main import Ebins
import os
folder = "./XmaxDists"  # a mappa elérési útja
files = [f for f in os.listdir(folder) if f.endswith(".txt")]  # csak .txt fájlok
n_files = len(files)

print(f"A mappában {n_files} db .txt fájl van.")



Ebins = Ebins
bins = 20
fig, axes = plt.subplots(4, 2)
axes = axes.flatten()
fig_width = 7 
fig_height = 9  
fig.set_size_inches(fig_width, fig_height)
plt.tight_layout()

for i in range(n_files): 
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

    skew_fit = gumbObj.skewness_fit(mu, beta)
    kurt_fit = gumbObj.kurtosis_fit(mu, beta)

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
    f'Skewness = {skew:.2f}\nKurtosis = {kurt:.2f}\nSkewness_fit = {skew_fit:.2f}\nKurtosis_fit = {kurt_fit:.2f}',
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize = 9
    )
    E_low = Ebins[i]
    E_high = Ebins[i+1]
    ax.set_title(f'Energy: {E_low:.2f} – {E_high:.2f} lg(E/eV)')
    ax.legend(fontsize=8)
    #StreamWriter
    with open(f"./FittingParameters/ParameterNumber_{i}.txt","w") as f:
        f.write(f"lgE\tmu\tmu_err\tbeta\tbeta_err\n")
        f.write(f"{E_low:.2f}_{E_high:.2f}\t{popt[0]}\t{perr[0]}\t{popt[1]}\t{perr[1]}\n")
plt.tight_layout(pad=2.0)
plt.show()
