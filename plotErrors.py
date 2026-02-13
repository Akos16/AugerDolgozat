import pandas as pd
import matplotlib.pyplot as plt

#Legend: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
# Fájl beolvasása
df = pd.read_csv(
    "./FittingParameters/ParameterNumbers.txt",
    sep=r"\s+",     
    na_values=["nan", "NaN"]
)

lgE = df["lgE"]

params = [
    ("mu", "mu_err"),
    ("beta", "beta_err"),
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
        label='err'
    )
    
    ax.set_ylabel(param)
    ax.set_title(f"{param} vs lgE")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize="medium", loc=2)
    if param == "beta":
        ax.legend(fontsize="medium", loc=1)
    # Skew és kurt referencia vonalak
    if param == "skew":
        ax.axhline(y=1.14, color='red', linestyle='--', linewidth=1.5, label='y=1.14')
        ax.legend(fontsize="medium", loc=2)
    if param == "kurt":
        ax.axhline(y=2.4, color='red', linestyle='--', linewidth=1.5, label='y=2.4')
        ax.legend(fontsize="medium", loc=2)
    
    #chi
    if param == "mu":
        for j in range(len(lgE)):
            text = f"χ²/ndf={df['chi2red'][j]:.2f}"  
            ax.text(
                lgE[j], y[j] + 0.05 * (max(y)-min(y)),  
                text,
                fontsize=8,
                ha='center'
            )


# X
for ax in axes[n_cols:]:  
    ax.set_xlabel("lgE")

plt.tight_layout()
plt.show()
