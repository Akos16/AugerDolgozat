import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

# Paraméterek
mu = 120.0
beta = 15.0
x0 = 150.0

# Tartomány és függvények
x = np.linspace(60, 220, 800)
pdf = gumbel_r.pdf(x, loc=mu, scale=beta)
cdf = gumbel_r.cdf(x, loc=mu, scale=beta)

# Keresett valószínűségek
p_leq = gumbel_r.cdf(x0, loc=mu, scale=beta)
p_gt = 1 - p_leq

# Ábra
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# PDF
ax[0].plot(x, pdf, color="tab:blue", lw=2, label="sűrűségfüggvény (PDF)")
ax[0].axvline(x0, color="tab:red", ls="--", label=f"x = {x0:g}")
ax[0].set_title("Gumbel-eloszlás PDF")
ax[0].set_xlabel("x")
ax[0].set_ylabel("f(x)")
ax[0].grid(alpha=0.3)
ax[0].legend()

# CDF
ax[1].plot(x, cdf, color="tab:green", lw=2, label="eloszlásfüggvény (CDF)")
ax[1].axvline(x0, color="tab:red", ls="--", label=f"x = {x0:g}")
ax[1].scatter([x0], [p_leq], color="black", zorder=3)
ax[1].annotate(f"P(X≤{x0:g}) = {p_leq:.3f}\nP(X>{x0:g}) = {p_gt:.3f}",
               xy=(x0, p_leq), xytext=(x0 + 8, p_leq - 0.2),
               arrowprops=dict(arrowstyle="->", lw=1))
ax[1].set_title("Gumbel-eloszlás CDF")
ax[1].set_xlabel("x")
ax[1].set_ylabel("F(x)")
ax[1].set_ylim(0, 1.02)
ax[1].grid(alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.show()
