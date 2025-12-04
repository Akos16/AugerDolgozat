import numpy as np

# Xmax értékek 710-től 950-ig 10-enként (példa)
xmax_values = np.arange(710, 951, 5)  # 710, 715, 720, ..., 950
n = len(xmax_values)

# Másik két oszlop (például 1-től n-ig és 1000-től 1000+n-1-ig)
szia_values = np.arange(1, n+1)
asd_values = np.arange(1000, 1000+n)

# Egyesítsük a három oszlopot
data = np.column_stack((szia_values, xmax_values, asd_values))

# Mentés txt-be
np.savetxt("data1.txt", data, fmt="%d", delimiter="\t", header="Szia\tXmax\tAsd", comments="")

print("adatok.txt elkészült!")