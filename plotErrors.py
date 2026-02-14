import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Legend: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
# Fájl beolvasása
df = pd.read_csv(
    "./FittingParameters/ParameterNumbers.txt",
    sep=r"\s+",     
    na_values=["nan", "NaN"]
)

lgE = df["lgE"]

params = [
    ("mean", "mean_err"),
    ("var", "var_err"),
    ("skew", "skew_err"),
    ("kurt", "kurt_err")
]

n_params = len(params)
n_rows = 2
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharex=True)
axes = axes.flatten()

for i, (param, err) in enumerate(params):
    ax = axes[i]
    y = df[param]
    yerr = df[err] if err is not None else None
    
    # Mérési pontok
    ax.errorbar(
        lgE,
        y,
        yerr=yerr,
        fmt='o' if yerr is not None else 's',
        elinewidth = 2,
        label='Auger data'
    )

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize="large", loc=2)
    ax.tick_params(axis='both', labelsize=14)

    if param == "mean":
        ax.set_ylabel(r"$\mu$", fontsize=16)
    if param == "var":
        ax.set_ylabel(r"$\sigma$", fontsize=16)
        ax.legend(fontsize="large", loc=1)
    # Skew és kurt referencia vonalak
    if param == "skew":
        ax.set_ylabel(r"$\gamma_1$", fontsize=16)
        ax.axhline(y=1.14, color='red', linestyle='--', linewidth=1.5, label=r"$\gamma_1(Gumbel)$=1.14")
        ax.legend(fontsize="large", loc=2)
    if param == "kurt":
        ax.set_ylabel(r"$\beta_2$", fontsize=16)
        ax.axhline(y=2.4, color='red', linestyle='--', linewidth=1.5, label=r"$\beta_2(Gumbel)$=2.4")
        ax.legend(fontsize="large", loc=2)
    
    #chi
    if param == "mean" and 1==0:
        for j in range(len(lgE)):
            text = f"χ²/ndf={df['chi2red'][j]:.2f}"  
            ax.text(
                lgE[j], y[j] + 0.05 * (max(y)-min(y)),  
                text,
                fontsize=8,
                ha='center'
            )
    
'''
ax_chi = axes[4]   
    ax_chi.plot(lgE, df["chi2red"], 'o')
    ax_chi.set_yticks(np.arange(0, 2, 0.2))
    ax_chi.set_xlabel("lgE")
    ax_chi.set_ylabel("χ²/ndf")
    ax_chi.set_title("χ²/ndf vs lgE")
    ax_chi.grid(True, linestyle='--', alpha=0.5)
    axes[5].axis("off")  
'''
# X
for ax in axes[n_cols:]:  
    ax.set_xlabel(r"$\log(E_{proton})$", fontsize=16)

plt.tight_layout()
plt.savefig("./figs/newest_simpler_estimator_parameters.png")
plt.show()