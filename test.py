import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from DataHandler import DataHandler
from Distributions import Gumbel
from main import Ebins
import os

# ===== fájlok =====
folder = "./XmaxDists"
files = [f for f in os.listdir(folder) if f.endswith(".txt")]
n_files = len(files)

gumbObj = Gumbel()

# ===== PRECOMPUTE =====
precomputed = []
y_max_global = 0

for i in range(n_files):

    filename = f"./XmaxDists/XmaxDist_Ebin{i}.txt"
    Xmax, Counts, CountsSqrt = DataHandler(filename).getData()

    mu0, beta0 = 700, 10
    a0 = np.max(Counts)

    sigma = np.sqrt(Counts)
    sigma[sigma == 0] = 1.0

    popt, _ = curve_fit(
        gumbObj.model,
        Xmax, Counts,
        p0=[mu0, beta0, a0],
        sigma=sigma,
        absolute_sigma=True
    )

    mu, beta, a = popt

    x_model = np.linspace(min(Xmax), max(Xmax), 300)
    y_model = gumbObj.model(x_model, mu, beta, a)

    y_max_global = max(y_max_global, np.max(Counts))

    precomputed.append({
        "x": Xmax,
        "y": Counts,
        "yerr": CountsSqrt,
        "x_model": x_model,
        "y_model": y_model,
        "E_low": Ebins[i],
        "E_high": Ebins[i+1]
    })

# ===== FIGURE =====
fig, axes = plt.subplots(4, 2, sharex=True, sharey=True)
axes = axes.flatten()
fig.set_size_inches(7, 9)

data_plots = []
fit_plots = []
err_plots = []
titles = []

for ax in axes:
    d, = ax.plot([], [], 'o', markersize=2, color='black')
    f, = ax.plot([], [], '-', color='red', linewidth=1.5)

    err = ax.errorbar([], [], yerr=[], fmt='none', ecolor='black', alpha=0.0)

    t = ax.set_title("", fontsize=8)

    ax.set_xlim(602, 949)
    ax.set_ylim(0, y_max_global * 1.2)

    data_plots.append(d)
    fit_plots.append(f)
    err_plots.append(err)
    titles.append(t)

fig.text(0.5, 0.02, r"$X_{\mathrm{max}}$ (g/cm²)", ha='center')
fig.text(0.04, 0.5, "Események száma", va='center', rotation='vertical')

fig.subplots_adjust(top=0.98, bottom=0.08, left=0.12, right=0.95)

# ===== ANIMÁCIÓ PARAMÉTEREK =====
frames_per_plot = 60
total_frames = n_files * frames_per_plot

def update(frame):

    artists = []

    plot_idx = frame // frames_per_plot
    local_frame = frame % frames_per_plot

    for i in range(n_files):

        if i > plot_idx:
            continue

        d = precomputed[i]

        titles[i].set_text(f"[{d['E_low']:.2f}, {d['E_high']:.2f})")

        # ===== FÁZISOK =====
        if i < plot_idx:
            # már kész subplot
            data_plots[i].set_data(d["x"], d["y"])
            fit_plots[i].set_data(d["x_model"], d["y_model"])

        else:
            # aktuális subplot

            # 1️⃣ pontok fokozatosan
            n_points = int(len(d["x"]) * local_frame / (frames_per_plot * 0.5))
            data_plots[i].set_data(d["x"][:n_points], d["y"][:n_points])

            # hibák együtt jelennek meg
            if n_points > 0:
                err_plots[i] = axes[i].errorbar(
    d["x"][:n_points],
    d["y"][:n_points],
    yerr=d["yerr"][:n_points],
    fmt='none',
    ecolor='black',
    alpha=0.5,
    elinewidth=0.5,   # 👈 EZ A LÉNYEG
    capsize=1         # 👈 kis "sapka" a végén
)
                

            # 2️⃣ fit rajzolódik balról jobbra
            if local_frame > frames_per_plot * 0.5:
                frac = (local_frame - frames_per_plot * 0.5) / (frames_per_plot * 0.5)
                n_fit = int(len(d["x_model"]) * frac)
                fit_plots[i].set_data(
                    d["x_model"][:n_fit],
                    d["y_model"][:n_fit]
                )

        artists.extend([data_plots[i], fit_plots[i], titles[i]])

    return artists

# ===== ANIMÁCIÓ =====
ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=30,
    blit=False   # fontos: errorbar miatt
)



plt.show()