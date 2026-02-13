import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit
import scipy.stats
import numpy.polynomial.polynomial as poly

LegendFontSize=20
LabelFontSize= 20
TickFontSize= 15

import os.path
from zipfile import ZipFile
def AugerLoad(fdir,file):
    """
    Loads a file from the auger open data release. Can be either in the local directory,
    in the parent directory or in the augeropendata directory.
    File is identified by it directory *fdir* and filename *file* and can be found in the directory
    or in a zip file.
    """
    for loc in [".","..","augeropendata"]:
        fname=os.path.join(loc,fdir,file)
        if os.path.isfile(fname):
            return open(fname)
        zname=os.path.join(loc,fdir+".zip")
        if os.path.isfile(zname):
            with ZipFile(zname) as myzip:
                return myzip.open(os.path.join(fdir,file))


data = pd.read_csv(AugerLoad('summary','dataSummary.csv'))

data['lgE'] = np.log10(data['fd_totalEnergy']) + 18 # units: lg(E/eV)

xmax_data = data[(data.fd_hdXmaxEye == 1)].copy() # copy so we can add columns later

grouped_xmax_data = xmax_data.groupby('id')
unique_xmax_data = xmax_data.drop_duplicates('id').set_index('id')

n_events = len(xmax_data)
n_unique = len(unique_xmax_data)

print(f'''Read {n_events} Xmax events of which {n_unique} are unique.
The total number of FD events (counting multi-eye events as multiple events) is {n_events}.''')

## Calculate weights: w = 1/uncertainty^2
xmax_data['fd_e_weight'] = 1 / np.square(xmax_data.fd_dtotalEnergy)
xmax_data['fd_xmax_weight'] = 1 / np.square(xmax_data.fd_dxmax)
## Calculate value * w
xmax_data['fd_e_weighted'] = xmax_data.fd_totalEnergy * xmax_data.fd_e_weight
xmax_data['fd_xmax_weighted'] = xmax_data.fd_xmax * xmax_data.fd_xmax_weight

# average of energies
sum_of_e_weights = grouped_xmax_data['fd_e_weight'].sum()
fd_avg_e = grouped_xmax_data['fd_e_weighted'].sum() / sum_of_e_weights
unique_xmax_data['fd_avg_lgE'] = np.log10(fd_avg_e*1e18)
fd_davg_e = 1 / np.sqrt(sum_of_e_weights)
fd_davg_lge = fd_davg_e / fd_avg_e / np.log(10.)
unique_xmax_data['fd_davg_lgE'] = fd_davg_lge

# average of Xmax
sum_of_xmax_weights = grouped_xmax_data['fd_xmax_weight'].sum()
fd_avg_xmax = grouped_xmax_data['fd_xmax_weighted'].sum() / sum_of_xmax_weights
unique_xmax_data['fd_avg_xmax'] = fd_avg_xmax
fd_davg_xmax = 1 / np.sqrt(sum_of_xmax_weights)
unique_xmax_data['fd_davg_xmax'] = fd_davg_xmax

lge_min = np.amin(unique_xmax_data['fd_avg_lgE'])
lge_max = np.amax(unique_xmax_data['fd_avg_lgE'])
print((int(lge_max*10)+1)/10.)
#Ebins = [17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.45, 18.8, 19.2, (int(lge_max*10)+1)/10.]
Ebins = [17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.8, (int(lge_max*10)+1)/10.]
#Ebins = [17.8, 17.9, 18.0, 18.1, 18.2, 18.4, (int(lge_max*10)+1)/10.]
#Ebins = [17.8, 18.2, 18.45, (int(lge_max*10)+1)/10.]
#Ebins = np.arange(17.8, lge_max + 0.1, 0.1)
print(f'The minimum energy in this data set is lg(E/eV) = {lge_min:7.2f}.')
print(f'The maximum energy in this data set is lg(E/eV) = {lge_max:7.2f}.')

def GetRawXmaxMoments(Ebins, lgEList, XmaxList):
    XmaxDistList = []
    for E1, E2 in zip(Ebins[:-1], Ebins[1:]):
        ECut = (E1 <= lgEList) & (lgEList < E2)
        XmaxDistList.append(XmaxList[ECut])

    return XmaxDistList

XmaxDistList = GetRawXmaxMoments(Ebins,unique_xmax_data['fd_avg_lgE'],unique_xmax_data['fd_avg_xmax'])

for iset in range(len(XmaxDistList)):
    counts, bin_edges = np.histogram(XmaxDistList[iset], bins=50, range=(480, 1080))
    with open(f"./XmaxDists/XmaxDist_Ebin{iset}.txt","w") as f:
        f.write(f"Xmax\tCounts\tCountsSqrt\n")
        for i in range(len(counts)):
            f.write(f"{bin_edges[i]}\t{counts[i]}\t{np.sqrt(counts[i])}\n")
