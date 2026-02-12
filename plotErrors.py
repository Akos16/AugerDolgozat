import pandas as pd
import matplotlib.pyplot as plt
# Fájl beolvasása
df = pd.read_csv(
    "./FittingParameters/ParameterNumbers.txt",
    sep=r"\s+",     
    na_values=["nan", "NaN"]
)

lgE = df["lgE"]
mu = df["mu"]
mu_err = df["mu_err"]
plt.figure(figsize=(7,5))

plt.errorbar(
    lgE,
    mu,
    yerr=mu_err,
    fmt='o'
)

plt.xlabel("lgE")
plt.ylabel("mu")


plt.show()