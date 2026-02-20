import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from DataHandler import DataHandler
from Distributions import Gumbel
from main import Ebins
import os

#Kigyűjti azokat a fájlokat a ./XmaxDists mappából, amelyeknek .txt a kirerjesztése
folder = "./XmaxDists"  
files = [f for f in os.listdir(folder) if f.endswith(".txt")]  
n_files = len(files) #For ciklushoz kell, mennyi fájl található a mappában

#Main.py-ban található Ebin-ek
Ebins = Ebins
bins = 25
#4 oszlop, 2 sor a plots, 1 plot 7 inch széles, 9 inch magas
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

#.txt helye, amelybe elmenti a kapott adatokat
filename1 = "./FittingParameters/ParameterNumbers.txt"
#ParameterNumbers.txt fejléce, az adatokat a for ciklusban tölti bele, minden sor más adatokat tartalmaz
with open(filename1, 'w') as f: 
    f.write("lgE\tmean\tmean_err\tvar\tvar_err\tskew\tskew_err\tkurt\tkurt_err\tchi2\tndf\tchi2red\n")
#for ciklus, hossza = XmaxDists-ben található txt-k számával
for i in range(n_files): 
    #XmaxDists-ben a txt-k 0-tól növekvő sorrendben vannak számozva, beolvassa az elsőt, mely a 0-ik txt, a for ciklus indexe 0-val indul
    filename = f"./XmaxDists/XmaxDist_Ebin{i}.txt"
    #A fájlbeolvasás getData() függvénynek a visszatérési értékeit elmentjük 3 numpy tömbbe, paraméternek a DataHandler az éppen aktuális txt-t várja
    #X tengely értékek, gyakoriságok, gyakoriságok statisztikai hibája
    Xmax, Counts, CountsSqrt = DataHandler(filename).getData()
    mu = 700 #mu kezdőérték
    beta = 10 #béta kezdőérték
    gumbObj = Gumbel() #Gumbel osztály példányosítása a Distributions.py fájlból
    x_data=Xmax #Xmax lesz az X tengely
    y_data=Counts #Counts, Az Xmax-ok száma lesz az y tengely
    a = np.max(Counts) #Amplitudó, y tengely legnagyobb adata

    sigma = np.sqrt(y_data) #y tengely adatainak négyzetgyöke a sigma
    sigma[sigma == 0] = 1.0 #ha a Counts-ban 0 fordulna elő, 1-et állít be
    
    #Gumbel osztályban található model() függvényt illeszti a mérési adatokra, paraméternek az x és y data-t várja
    #nemlineáris legkisebb négyzetek módszerével (scipy.optimize.curve_fit)
    #p0, kezdeti paraméterbecslési értékek
    #siga, mérési hiba, ha absolute_sigma = True akkor abszolút hibaként kezeli
    #popt, illesztett optimális paraméterek
    #pcov, paraméterek kovariancia-mátrixa (ez adja a paraméter hibákat)
    popt, pcov = curve_fit(gumbObj.model, x_data, y_data, p0=[mu, beta, a], sigma=sigma, absolute_sigma=True)

    #moments_with_errors() függvény visszatérési értéke, paraméternek a txt oszlopait várja
    #mean = átlag, mean_err = átlag hibája, var = variancia, var_error = variancia hiba
    #skew = ferdeség, skew_err = ferdeség hibája, kurt = csúcsosság, kurt_err = csúcsosság hibája
    mean, mean_err, var, var_err, skew,  skew_err, kurt, kurt_err = moments_with_errors(Xmax, Counts, CountsSqrt)

    #Gumbel osztályban található fittelt skewness és kurtosis visszatérési értékei, paraméternek a mu-t és beta-t varja
    skew_fit, mean1, var1 = gumbObj.skewness_fit(mu, beta)
    kurt_fit = gumbObj.kurtosis_fit(mu, beta)

    #Kovariancia-mátrix főátlójából számítja a paraméterek alap hibáit
    perr = np.sqrt(np.diag(pcov))

    #Optimális paraméterek mentése, mu = hely, beta = skala, a = amplitudó
    mu, beta, a = popt

    #folytonos x tengely a modell kirajzoláshoz, adatok min - max a range, 100 ponttal reprezentálva
    x_model = np.linspace(min(x_data), max(x_data), 100)

    #Model értékek kiszámítása az illesztett paraméterekkel
    #Simított ilesztett görbe
    y_model = gumbObj.model(x_model, mu, beta, a)

    #Varianciák és kovarianciák összeadása = modellérték bizonytalanság
    dy2 = perr[0] + perr[1] + pcov[0][1] + perr[2] + pcov[1][2] + pcov[0][2]
    #teljes bizonytalanság (szórás) a variancia négyzetgyöke
    dy = np.sqrt(dy2)

    #jelenlegi sublot, index értéke, az i-ik számú txt-t is jelöli
    ax = axes[i]

    #x tengely, y tengely, statisztikai hiba
    x, y, yerr = Xmax, Counts, CountsSqrt

    #Modell bizonytalansági sáv, fill_between a görbe körüli dy-t tölti ki
    #0,3 áttetszőség, zorder = kirajzolási sorrend, kisebb háttérben
    ax.fill_between(x_model, y_model - dy, y_model + dy, alpha=0.3, zorder=1, label='Model bizonytalanság')

    #Illesztett Gumbel-eloszlás kirajzolása folytonos görbeként
    #Piros szín, 2-es vastagságú vonal, zorder = 2, bizonytalansági sáv fülé kerül
    ax.plot(x_model, y_model, color='red', linewidth=2, zorder=2, label='Gumbel illesztés')

    #Mért adatok kirajzolása hibasávokkal
    #Pont jelölés, markersize = kisméretű pontok, capsize = hibavonal végének mérete
    #Elinewidth = hibavonal vastagsága, zorder = 3, legfelül jelenik meg
    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Mért adat')
    
    #Khi-négyzet inícializálása
    chi2 = 0

    #Összes mérési ponton végig megy a for ciklus
    for y in range(len(y_data)):
        # Reziduális: mért érték - modellérték
        residuals = y_data[y] - gumbObj.model(x_data[y], popt[0], popt[1], popt[2])
        #Csak akkor számol, ha a mérési érték nem 0
        if(yerr[y] > 0):
            #Khi-négyzet definíció szerinti összegzés:
            #(reziduális / hiba)^2
            chi2 += np.sum((residuals / yerr[y])**2)
    
    #Szabadsági fok (number of degrees of freedom)
    #N adatpont - p illesztett paraméter
    ndf = len(y_data) - len(popt)
    
    #Redukált khi-négyzet
    chi2_red = chi2 / ndf
    #Eredmények kiírása
    print("chi2 =", chi2)
    print("ndf =", ndf)
    print("chi2/ndf =", chi2_red)

    #Adott txt-ben található Ebin tartomány
    E_low = Ebins[i]
    E_high = Ebins[i+1]
    #Ebin tartomány kiírása
    energy_label = rf"$E_{{\mathrm{{bin}}}} = [{E_low:.2f}, {E_high:.2f})$"

    #Legend megjelenítése, cím = energy_label, 7-es betűméret, 8-as cím méret, 
    ax.legend(
        title=energy_label,
        fontsize=7,
        title_fontsize=8,
        frameon=False #keretre legyen-e rajzolva a felirat
    )
    
    #Ebinek átlaga
    E_Avg = (E_low + E_high) / 2
    #StreamWriter, mért adatok kiíratása a fent definiált txt-be (ParameterNumbers.txt)
    #Forciklus i-ik eleme = 1 sor
    #Tabulátorral elválasztva
    #E_Avg = átlagenergia, mean = eloszlás átlaga, mean_err = átlag hibája
    #np.sqrt(var) = szórás, var_err / (2 * sqrt(var)) = szórás hibája
    #skew = ferdeség, skew_err = ferdeség hibája, kurt = csúcsosság, kurt_err = csúcsosság hibája
    #chi2 = khi-négyzet, ndf = szabadsági fok, chi2_red = redukált khi-négyzet
    #\n sor vége, új sor
    with open(filename1, 'a') as f:
        f.write(f"{E_Avg}\t{mean}\t{mean_err}\t{np.sqrt(var)}\t{var_err/2./np.sqrt(var)}\t{skew}\t{skew_err}\t{kurt}\t{kurt_err}\t{chi2}\t{ndf}\t{chi2_red}\n")

#Összes subplot
#enumerate(axes): i az index, ax az aktuális tengely objektum
for i, ax in enumerate(axes):
    #Meghatározzuk a subplot rácsban elfoglalt pozíciót, (feltételezve 2 oszlopos elrendezést)
    row = i // 2
    col = i % 2
    #A felső sorok esetén elrejtjük az x-tengely számfeliratait, hogy az ábra áttekinthetőbb legyen.
    if row < 3:
        ax.tick_params(labelbottom=False)
    #A jobb oldali oszlop esetén elrejtjük a y-tengely számfeliratait, hogy ne ismétlődjenek feleslegesen.
    if col > 0:
        ax.tick_params(labelleft=False)
#Az összes subplot x-tengelyének egységes beállítása
#Az Xmax tartomány 600–950 g/cm² között jelenik meg
for ax in axes:
    ax.set_xlim(600, 950)
#Az alsó sor subplotjain az x-tengely osztásközeinek beállítása
#50 g/cm² lépésközzel jelennek meg a tick-ek
for ax in axes[-2:]:
    ax.set_xticks(np.arange(600, 951, 50))

# Alsó x-tengely felirat kicsit feljebb 
fig.text(0.525, 0.02, r"$X_{\mathrm{max}}$ (légköri maximum, g/cm²)", ha='center')

#y-tengely felirat
fig.text(0.04, 0.5, "Események száma", va='center', rotation='vertical')
#Margók, felíratok ne lógjanak bele a számokba
fig.subplots_adjust(top=0.99, bottom=0.07, left=0.12, right=0.95)
#Ábra mentése a ./figs/ mappába, newest_simpler_curvefit néven, png formátumban
plt.savefig("./figs/newest_simpler_curvefit.png")
#Ábra megjelenítése
plt.show()