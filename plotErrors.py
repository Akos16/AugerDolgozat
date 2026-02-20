import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "./FittingParameters/ParameterNumbers.txt",
    sep=r"\s+",
    na_values=["nan", "NaN"]
)
df1 = pd.read_csv(
    "./FittingParameters/Full_data.txt",
    sep=r"\s+",
    na_values=["nan", "NaN"]
)
lgE = df["lgE"]
print(df1.columns)
params = [
    ("mean", "mean_err"),
    ("var", "var_err"),
    ("skew", "skew_err"),
    ("kurt", "kurt_err")
]

for param, err in params:

    plt.figure(figsize=(8, 6))

    y = df[param]
    yerr = df[err] if err is not None else None

    plt.errorbar(
        lgE,
        y,
        yerr=yerr,
        fmt='o' if yerr is not None else 's',
        elinewidth=2,
        label='Auger adat'
    )
    if param in ["mean", "var"]:
        plt.errorbar(
            df1["lgE"],
            df1[param],
            yerr=df1[err],
            fmt='o' if yerr is not None else 's',
            elinewidth=2,
            label='Auger eredeti teljes adat'
        )

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', labelsize=14)

    if param == "mean":
        plt.ylabel(r"$\mu$", fontsize=16)

    if param == "var":
        plt.ylabel(r"$\sigma$", fontsize=16)

    if param == "skew":
        plt.ylabel(r"$\gamma_1$", fontsize=16)
        plt.axhline(y=1.14, color='red', linestyle='--', linewidth=1.5,
                    label=r"$\gamma_1(Gumbel)$=1.14")

    if param == "kurt":
        plt.ylabel(r"$\beta_2$", fontsize=16)
        plt.axhline(y=2.4, color='red', linestyle='--', linewidth=1.5,
                    label=r"$\beta_2(Gumbel)$=2.4")

    plt.xlabel(r"$\log(E_{proton})$", fontsize=16)
    if param != "var":
        plt.legend(fontsize="large", loc=2)
    else:
        plt.legend(fontsize="large")
    plt.tight_layout()
    plt.savefig(f"./figs/{param}_vs_lgE.png")
    plt.show()

plt.figure(figsize=(8, 6))

plt.plot(lgE, df["chi2red"], 'o', label="χ²/ndf")
plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5)

plt.yticks(np.arange(0, 2, 0.2))
plt.xlabel("lgE", fontsize=16)
plt.ylabel("χ²/ndf", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("./figs/chi2_vs_lgE.png")
plt.show()