import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from DataHandler import DataHandler
from Distributions import Gumbel
from main import Ebins
import os


folder = "./XmaxDists"  
files = [f for f in os.listdir(folder) if f.endswith(".txt")]  
n_files = len(files)


Ebins = Ebins
bins = 25
fig, axes = plt.subplots(4, 2, sharex=True, sharey=True)
axes = axes.flatten()
fig_width = 7
fig_height = 9


fig.set_size_inches(fig_width, fig_height)
plt.tight_layout()

def moments_from_prob(bin_centers, hist_counts):
    P = hist_counts / np.sum(hist_counts)   
    n_data = int(np.sum(hist_counts))       

    mean = np.sum(bin_centers * P)

    mu2 = np.sum(((bin_centers - mean) ** 2) * P)
    mu3 = np.sum(((bin_centers - mean) ** 3) * P)
    mu4 = np.sum(((bin_centers - mean) ** 4) * P)

    skew = mu3 / mu2**1.5
    excess_kurt = mu4 / mu2**2 - 3

    skew_err = np.sqrt(6 / n_data)
    kurt_err = np.sqrt(24 / n_data)

    return mean, mu2, skew, excess_kurt


def moments_with_errors(x_or_edges, counts, count_err,
                        ntoy=5000, seed=0):
    """
    Returns:
    mean, mean_err
    variance, var_err
    skew, skew_err
    kurtosis, kurt_err
    """

    rng = np.random.default_rng(seed)

    counts = np.asarray(counts, float)
    errs   = np.asarray(count_err, float)

    # ---- baseline values ----
    mean0, var0, skew0, kurt0 = moments_from_prob(x_or_edges, counts)

    mean_list = []
    var_list  = []
    skew_list = []
    kurt_list = []

    for _ in range(ntoy):

        # Gaussian toy histogram
        toy = counts + rng.normal(0.0, errs)

        # enforce positivity
        toy = np.clip(toy, 0.0, None)

        if np.sum(toy) <= 0:
            continue

        m, v, s, k = moments_from_prob(x_or_edges, toy)

        mean_list.append(m)
        var_list.append(v)
        skew_list.append(s)
        kurt_list.append(k)

    mean_err = np.std(mean_list, ddof=1)
    var_err  = np.std(var_list,  ddof=1)
    skew_err = np.std(skew_list, ddof=1)
    kurt_err = np.std(kurt_list, ddof=1)

    return (mean0, mean_err,
            var0, var_err,
            skew0, skew_err,
            kurt0, kurt_err)
filename1 = "./FittingParameters/ParameterNumbers.txt"
with open(filename1, 'w') as f:
    f.write("lgE\tmean\tmean_err\tvar\tvar_err\tskew\tskew_err\tkurt\tkurt_err\tchi2\tndf\tchi2red\n")

for i in range(n_files): 
    filename = f"./XmaxDists/XmaxDist_Ebin{i}.txt"
    Xmax, Counts, CountsSqrt = DataHandler(filename).getData()
    mu = 700
    beta = 10
    gumbObj = Gumbel()
    x_data=Xmax
    y_data=Counts
    a = np.max(Counts)

    sigma = np.sqrt(y_data)
    sigma[sigma == 0] = 1.0
    popt, pcov = curve_fit(gumbObj.model, x_data, y_data, p0=[mu, beta, a], sigma=sigma, absolute_sigma=True)

    mean, mean_err, var, var_err, skew,  skew_err, kurt, kurt_err = moments_with_errors(Xmax, Counts, CountsSqrt)
    #eloszlás, extrémérték eloszlás, dobozos véletlen mindig max és ábrázolás
    #Augernál mi az xmax, mit mér, minek az xmaxot, és hogy ez egy extrém 
    skew_fit, mean1, var1 = gumbObj.skewness_fit(mu, beta)
    kurt_fit = gumbObj.kurtosis_fit(mu, beta)

    perr = np.sqrt(np.diag(pcov))

    mu, beta, a = popt
    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = gumbObj.model(x_model, mu, beta, a)

    dy2 = perr[0] + perr[1] + pcov[0][1] + perr[2] + pcov[1][2] + pcov[0][2]
    dy = np.sqrt(dy2)
    ax = axes[i]
    x, y, yerr = Xmax, Counts, CountsSqrt

    ax.fill_between(x_model, y_model - dy, y_model + dy, alpha=0.3, zorder=1, label='Model uncertainity')

    ax.plot(x_model, y_model, color='red', linewidth=2, zorder=2, label='Gumbel fit')

    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Measured data')
    
    chi2 = 0
    for y in range(len(y_data)):
        residuals = y_data[y] - gumbObj.model(x_data[y], popt[0], popt[1], popt[2])
        if(yerr[y] > 0):
            chi2 += np.sum((residuals / yerr[y])**2)
        
    ndf = len(y_data) - len(popt)

    chi2_red = chi2 / ndf
    print("chi2 =", chi2)
    print("ndf =", ndf)
    print("chi2/ndf =", chi2_red)

    E_low = Ebins[i]
    E_high = Ebins[i+1]
    energy_label = rf"$E_{{\mathrm{{bin}}}} = [{E_low:.2f}, {E_high:.2f})$"

    ax.legend(
        title=energy_label,
        fontsize=7,
        title_fontsize=8,
        frameon=False
    )
    
    E_Avg = (E_low + E_high) / 2
    #StreamWriter
    with open(filename1, 'a') as f:
        f.write(f"{E_Avg}\t{mean}\t{mean_err}\t{np.sqrt(var)}\t{var_err/2./np.sqrt(var)}\t{skew}\t{skew_err}\t{kurt}\t{kurt_err}\t{chi2}\t{ndf}\t{chi2_red}\n")

for i, ax in enumerate(axes):
    row = i // 2
    col = i % 2
    if row < 3:
        ax.tick_params(labelbottom=False)
    if col > 0:
        ax.tick_params(labelleft=False)
#X
for ax in axes:
    ax.set_xlim(600, 950)

for ax in axes[-2:]:
    ax.set_xticks(np.arange(600, 951, 50))

# Alsó x-tengely felirat kicsit feljebb 
fig.text(0.525, 0.02, "Xmax (g/cm²)", ha='center')

# Bal oldali y-tengely felirat
fig.text(0.04, 0.5, "Counts", va='center', rotation='vertical')

fig.subplots_adjust(top=0.99, bottom=0.07, left=0.12, right=0.95)
plt.savefig("./figs/newest_simpler_curvefit.png")
plt.show()