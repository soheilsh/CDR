import numpy as np
from scipy import optimize
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import Symbol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import bisect
import six 
from six.moves import zip
import csv
from numpy import genfromtxt
import xlsxwriter as xlwt
from matplotlib.ticker import FuncFormatter

# ========================================== parameters ========================================== #
gamma = 0.5                                     # Share of children's welbeing in Utility function of parents
alpha = 0.55/2                                  # Agricultural share in Consumption function
eps = 0.5                                       # Elasticity of substitution in Consumption function
T = 6                                           # Time horizon
scl = 1
pscl = 20

mcU1 = 0.5                                      # Migration cost for unskilled labor in country 1 as afraction of tu1
mcS1 = 0.5                                      # Migration cost for skilled labor in country 1 as afraction of ts1
mcU2 = 0.5                                      # Migration cost for unskilled labor in country 2 as afraction of tu2
mcS2 = 0.5                                      # Migration cost for skilled labor in country 2 as afraction of ts2

# ========================================== RCP Scenarios =========================================== # 
RCP = ([882, 998, 1063, 1060, 1034, 1007], [882, 1001, 1139, 1266, 1318, 1335], [882, 995, 1113, 1286, 1518, 2476], [882, 1049, 1271, 1665, 2183, 2785])
# ========================================== Temperature =========================================== #
# Pole temperature 0 and Equator temperature 28
# Temp = - 28/(pi/2) * (lat - pi/2) 
mu1 = 0.21                                  #in Desmet it is 0.0003 but they report it after dividing by 1000/pi/2
mu2 = 0.5
mu3 = 0.0238

# ========================================== Damages =========================================== #
# D = g0 + g1 * T + g2 * T^2

# Agricultural parameters
g0a = -2.24
g1a = 0.308
g2a = -0.0073

# Manufacturing parameters
g0m = 0.3
g1m = 0.08
g2m = -0.0023

# ========================================== Variables =========================================== #                    

# == temperature == #
Temp1 = np.zeros((T, 4))                     # Temperature (Country 1)
Temp2 = np.zeros((T, 4))                     # Temperature (Country 2)

# == Age matrix == #
nu1 = np.zeros((T, 4))                       # number of unskilled children (Country 1)
ns1 = np.zeros((T, 4))                       # number of skilled children (Country 1)
L1 = np.zeros((T, 4))                        # Number of unskilled parents (Country 1)
H1 = np.zeros((T, 4))                        # Number of skilled parents (Country 1)
h1 = np.zeros((T, 4))                        # Ratio of skilled to unskilled labor h=H/L (Country 1)
hn1 = np.zeros((T, 4))                       # Ratio of skilled to unskilled children h=ns/nu (Country 1)
N1 = np.zeros((T, 4))                        # Adult population (Country 1)
Pop1 = np.zeros((T, 4))                      # Total population (Country 1)

nu2 = np.zeros((T, 4))                       # number of unskilled children (Country 2)
ns2 = np.zeros((T, 4))                       # number of skilled children (Country 2)
L2 = np.zeros((T, 4))                        # Number of unskilled parents (Country 2)
H2 = np.zeros((T, 4))                        # Number of skilled parents (Country 2)
h2 = np.zeros((T, 4))                        # Ratio of skilled to unskilled labor h=H/L (Country 2)
hn2 = np.zeros((T, 4))                       # Ratio of skilled to unskilled children h=ns/nu (Country 2)
N2 = np.zeros((T, 4))                        # Adult population (Country 1)
Pop2 = np.zeros((T, 4))                      # Total population (Country 1)

# == Prices == #
pa1 = np.zeros((T, 4))                       # Pice of AgricuLtural good (Country 1)
pm1 = np.zeros((T, 4))                       # Pice of Manufacturing good (Country 1)
pr1 = np.zeros((T, 4))                       # Relative pice of Manufacturing to Agricultural goods (Country 1)

pa2 = np.zeros((T, 4))                       # Pice of AgricuLtural good (Country 2)
pm2 = np.zeros((T, 4))                       # Pice of Manufacturing good (Country 2)
pr2 = np.zeros((T, 4))                       # Relative pice of Manufacturing to Agricultural goods (Country 2)

# == Wages == #
wu1 = np.zeros((T, 4))                       # Wage of unskilled labor (Country 1)
ws1 = np.zeros((T, 4))                       # Wage of skilled labor (Country 1)
wr1 = np.zeros((T, 4))                       # Wage ratio of skilled to unskilled labor (Country 1)

wu2 = np.zeros((T, 4))                       # Wage of unskilled labor (Country 2)
ws2 = np.zeros((T, 4))                       # Wage of skilled labor (Country 2)
wr2 = np.zeros((T, 4))                       # Wage ratio of skilled to unskilled labor (Country 2)

wrs = np.zeros((T, 4))                       #  Wage ratio of skilled labor Country 2 to Country 1
wru = np.zeros((T, 4))                       #  Wage ratio of unskilled labor Country 2 to Country 1

# == Technology == #
Aa1 = np.zeros((T, 4))                       # Technological growth function for Agriculture (Country 1)
Am1 = np.zeros((T, 4))                       # Technological growth function for Manufacurng (Country 1)
Ar1 = np.zeros((T, 4))                       # ratio of Technology in Manufacurng to Agriculture (Country 1)

Aa2 = np.zeros((T, 4))                       # Technological growth function for Agriculture (Country 2)
Am2 = np.zeros((T, 4))                       # Technological growth function for Manufacurng (Country 2)                                     # Technological growth rate for Manufacurng (Country 1)                  
Ar2 = np.zeros((T, 4))                       # ratio of Technology in Manufacurng to Agriculture (Country 2)

# == Output == #
Y1 = np.zeros((T, 4))                        # Total output (Country 1)
Ya1 = np.zeros((T, 4))                       # AgricuLtural output (Country 1)
Ym1 = np.zeros((T, 4))                       # Manufacturing output (Country 1)
Yr1 = np.zeros((T, 4))                       # Ratio of Manufacturing output to Agricultural output (Country 1)
Yp1 = np.zeros((T, 4))                       # Output per capita (Country 1)
YA1 = np.zeros((T, 4))                       # Output per adult (Country 1)

Y2 = np.zeros((T, 4))                        # Total output (Country 2)
Ya2 = np.zeros((T, 4))                       # AgricuLtural output (Country 2)
Ym2 = np.zeros((T, 4))                       # Manufacturing output (Country 2)
Yr2 = np.zeros((T, 4))                       # Ratio of Manufacturing output to Agricultural output (Country 2)
Yp2 = np.zeros((T, 4))                       # Output per capita (Country 2)
YA2 = np.zeros((T, 4))                       # Output per adult (Country 2)

# == Output == #
Da1 = np.zeros((T, 4))                       # AgricuLtural damage (Country 1)
Dm1 = np.zeros((T, 4))                       # Manufacturing damage (Country 1)
Dr1 = np.zeros((T, 4))                       # Ratio of Manufacturing damages to Agricultural damages (Country 1)

Da2 = np.zeros((T, 4))                       # AgricuLtural damage (Country 2)
Dm2 = np.zeros((T, 4))                       # Manufacturing damage (Country 2)
Dr2 = np.zeros((T, 4))                       # Ratio of Manufacturing damages to Agricultural damages (Country 2)

# == Consumption == #
cau1 = np.zeros((T, 4))                      # consumption of agricultural good unskilled (Country 1)
cas1 = np.zeros((T, 4))                      # consumption of agricultural good skilled (Country 1)
cmu1 = np.zeros((T, 4))                      # consumption of manufacturing good unskilled (Country 1)
cms1 = np.zeros((T, 4))                      # consumption of manufacturing good skilled (Country 1)
cu1 = np.zeros((T, 4))                       # consumption of all goods unskilled (Country 1)
cs1 = np.zeros((T, 4))                       # consumption of all goods skilled (Country 1)

cau2 = np.zeros((T, 4))                      # consumption of agricultural good unskilled (Country 2)
cas2 = np.zeros((T, 4))                      # consumption of agricultural good skilled (Country 2)
cmu2 = np.zeros((T, 4))                      # consumption of manufacturing good unskilled (Country 2)
cms2 = np.zeros((T, 4))                      # consumption of manufacturing good skilled (Country 2)
cu2 = np.zeros((T, 4))                       # consumption of all goods unskilled (Country 2)
cs2 = np.zeros((T, 4))                       # consumption of all goods skilled (Country 2)

# == Migration == #
bs12 = np.zeros((T, 4))                       # migration flow rate of skilled labor (from country 1 to country 2)
bu12 = np.zeros((T, 4))                       # migration flow rate of unskilled labor (from country 1 to country 2)
bs21 = np.zeros((T, 4))                       # migration flow rate of skilled labor (from country 2 to country 1)
bu21 = np.zeros((T, 4))                       # migration flow rate of unskilled labor (from country 2 to country 1)

Ms12 = np.zeros((T, 4))                       # migration flow level of skilled labor  (from country 1 to country 2)
Mu12 = np.zeros((T, 4))                       # migration flow level of unskilled labor  (from country 1 to country 2)
Ms21 = np.zeros((T, 4))                       # migration flow level of skilled labor  (from country 2 to country 1)
Mu21 = np.zeros((T, 4))                       # migration flow level of unskilled labor  (from country 2 to country 1)

dffx11 = np.zeros((T, 4))
dffx22 = np.zeros((T, 4))
dffx1 = np.zeros((T, 4))
dffx2 = np.zeros((T, 4))

PwsM1 = np.zeros((T, 4))
PwuM1 = np.zeros((T, 4))
PwsM2 = np.zeros((T, 4))
PwuM2 = np.zeros((T, 4))
# ============================================== Country Calibration ============================================== #
# hx: Ratio of skilled to unskilled labor in 2100 hx = Hx/Lx
# popgx: population growth rate in 2100
# nux: number of unskilled children per parent in 2100
# nsx: number of skilled children per parent in 2100
# Arx: Ratio of technology in manufacturing to technology in agriculture in 2100
# H0: skilled labor in 1980 (milion people)
# L0: unskilled labor in 1980 (milion people)
# h0: Ratio of skilled to unskilled labor in 1980 h0 = H0/L0
# nu0: number of unskilled children per parent in 1980
# ns0: number of skilled children per parent in 1980
# Am0: technology in manufacturing in 1980
# Aa0: technology in agriculture in 1980
# Ar0: Ratio of technology in manufacturing to technology in agriculture in 1980
# N0: Total labor in 1980 (milion people)
# Y0: GDP in 1980 (bilion 1990 GK$)
# C0: population of children in 1980
# r0: ratio of ts/tu
# lat, latid: latitude
# =========== AFRICA ============ #4.5709,25,45,65
Ndata1 = [462.103, 785.970, 1211.090, 1661.808, 1972.735, 2174.992]
hdata1 = [0.16, 0.32, 0.58, 0.92, 1.47, 2.32]

Amgr1 = 0.00
H001 = 19.303
L001 = 250.936
Y001 = 0.731
latid1 = 5
Init1 = [H001, L001, Y001, latid1]

# =========== W. Europe ============ #4.5709,25,45,65
Ndata2 = [150.094, 163.112, 176.665, 177.708, 180.390, 176.952]
hdata2 = [1.73, 3.00, 4.80, 8.56, 13.50, 20.75]

Amgr2 = 0.02
H002 = 63.018
L002 = 73.237
Y002 = 2.377
latid2 = 45
Init2 = [H002, L002, Y002, latid2]

# =========== PARAGUAY ============ #   
#Ndata = [3.20, 5.02, 6.51, 7.51, 7.76, 7.45]
#hdata = [0.24, 0.51, 0.88, 1.48, 2.54, 4.73]
#
#H0 = 0.220
#L0 = 1.569
#Y0 = 10.356
#latid = 23.4425
# =========== FINLAND ============ #   
#Ndata = [4.24, 4.69, 4.82, 4.80, 4.91, 4.98]
#hdata = [1.41, 3.42, 3.94, 5.38, 7.40, 9.63]
#
#H0 = 1.431
#L0 = 2.386
#Y0 = 61.781
#latid = 60.4802
# =========== DENMARK ============ #   
#Ndata2 = [4.37, 4.85, 5.06, 5.34, 5.69, 5.74]
#hdata2 = [1.56, 2.50, 4.13, 6.55, 8.44, 13.00]
#
#H002 = 1.741
#L002 = 2.304
#Y002 = 77.975
#latid2 = 55.6761
#Init2 = [H002, L002, Y002, latid2]
# =========== ESTONIA ============ #   
#Ndata = [1.14, 1.09, 1.02, 0.92, 0.84, 0.79]
#hdata = [2.12, 5.46, 9.75, 13.33, 16.40, 20.75]
#
#H0 = 0.544
#L0 = 0.602
#Y0 = 7.347
#latid = 58.5953
### ============================================== Model Calibration ============================================== #
def Calib(hdatax, Ndatax, INITx, Amgrx):
    [H00, L00, Y00, latid] = INITx
    hx = hdatax[5]
    popgx = (Ndatax[5] - Ndatax[4])/Ndatax[4]
    N0 = Ndatax[0]
    
    nux = (1 + popgx) / (1 + hx)
    nsx = (1 + popgx) * hx / (1 + hx)
    
    N00 = H00 + L00
    h00 = H00/L00
    
    lat = latid * np.pi/180
    Temp0 = - 28/(np.pi/2) * (lat - np.pi/2)
    Da0 = max(0.001, g0a + g1a * Temp0 + g2a * Temp0**2)
    Dm0 = max(0.001, g0m + g1m * Temp0 + g2m * Temp0**2)
    Dr0 = Dm0/Da0
    
    Con1 = RCP[0][1] - RCP[0][0]
    Temp1 = Temp0 + mu1 * Con1**mu2 * (1 - mu3 * Temp0)
    Da1 = max(0.001, g0a + g1a * Temp1 + g2a * Temp1**2)
    Dm1 = max(0.001, g0m + g1m * Temp1 + g2m * Temp1**2)
    Dr1 = Dm1/Da1
    
    Conx = RCP[0][T - 1] - RCP[0][0]
    Tempx = Temp0 + mu1 * Conx**mu2 * (1 - mu3 * Temp0)
    Dax = max(0.001, g0a + g1a * Tempx + g2a * Tempx**2)
    Dmx = max(0.001, g0m + g1m * Tempx + g2m * Tempx**2)
    Drx = Dmx/Dax
    
    def hsolve(rx):
        Arx = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(rx) - np.log(hx)) /(1 - eps) - np.log(Drx))
        Ar0 = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(rx) - np.log(h00)) /(1 - eps) - np.log(Dr0))
        Argx = np.exp((np.log(Arx/Ar0))/((2100 - 1980)/20)) - 1
        h1 = np.exp(eps * (np.log((1 - alpha)/alpha) - np.log(rx) - (1 - eps)/eps * np.log((Dr1) * Ar0 * (1 + Argx))))
        tu1x = gamma * (1 + h1) / (h1 * rx + 1) * N00/N0
        tu2x = gamma / (rx * nsx + nux)
        return tu1x - tu2x
    
    r0 = bisect(hsolve, 100, 0.01)
    tu = gamma / (r0 * nsx + nux)
    ts = tu * r0
    
    Ar00 = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(r0) - np.log(h00)) /(1 - eps) - np.log(Dr0))
    Am00 = Y00/((alpha * (L00 * Da0 / Ar00)**((eps - 1)/eps) + (1 - alpha) * (H00 * Dm0)**((eps - 1)/eps))**(eps/(eps - 1)))
    Aa00 = Am00/Ar00
    
    Arx = np.exp((eps * np.log((1 - alpha)/alpha) - eps * np.log(r0) - np.log(hx)) /(1 - eps) - np.log(Drx))
    Arg = np.exp((np.log(Arx/Ar00))/((2100 - 1980)/20)) - 1
    
    Amg = (1 + Amgrx)**20 - 1
    Aag = (1 + Amg)/(1 + Arg) - 1
    
    Ar0 = Ar00 * (1 + Arg)
    Aa0 = Aa00 * (1 + Aag)
    Am0 = Am00 * (1 + Amg)
    
    h0 = np.exp(eps * (np.log((1 - alpha)/alpha) - np.log(r0) - (1 - eps)/eps * np.log(Dr0 * Ar0)))
    
    nu0 = gamma / (h0 * ts + tu)
    ns0 = nu0 * h0
    
    L0 = N0 / (1 + h0)
    H0 = L0 * h0
    
    Ya0 = Aa0 * L0 * Da0
    Ym0 = Am0 * H0 * Dm0
    Yr0 = Ym0 / Ya0
    Y0 = (alpha * Ya0**((eps -1)/eps) + (1 - alpha) * Ym0**((eps -1)/eps))**(eps/(eps - 1))
    
    pr0 = (Yr0)**(-1/eps) * alpha / (1 - alpha)
    
    YA0 = Ya0 / N0
    cmu0 = Ym0 / (H0 * r0 + L0)
    cms0 = cmu0 * r0
    cau0 = Ya0 / (H0 * r0 + L0)
    cas0 = cau0 * r0    
    cu0 = (alpha * cau0**((eps - 1)/eps) + (1 - alpha) * cmu0**((eps - 1)/eps))**(eps/(eps - 1))
    cs0 = (alpha * cas0**((eps - 1)/eps) + (1 - alpha) * cms0**((eps - 1)/eps))**(eps/(eps - 1))
    wu0 = cu0 / (1 - gamma)
    ws0 = cs0 / (1 - gamma)
    pa0 = wu0 / (Da0 * Aa0)
    pm0 = ws0 / (Dm0 * Am0)
    wr0 = ws0/wu0
    Outputx = [N0, Temp0, h0, Da0, Dm0, Dr0, H0, L0, Y0, YA0, Ya0, Ym0, Yr0, Aa0, Am0, cmu0, cms0, cau0, cas0, cu0, cs0, wu0, ws0, wr0, pa0, pm0, pr0]
    Ratex = [Aag, Amg, tu, ts]
    return (Outputx, Ratex)

# ============================================== Transition Function ============================================== #
global bMs12, bMu12, bMs21, bMu21
global tsM1, tuM1, NM1, AaM1, AmM1, DaM1, DmM1, DrM1, ArM1
global tsM2, tuM2, NM2, AaM2, AmM2, DaM2, DmM2, DrM2, ArM2

for j in range(4):
    [Output1, Rate1] = Calib(hdata1, Ndata1, Init1, Amgr1)
    [Aag1, Amg1, tu1, ts1] = Rate1
    [N1[0, j], Temp1[0, j], h1[0, j], Da1[0, j], Dm1[0, j], Dr1[0, j], H1[0, j], L1[0, j], Y1[0, j], YA1[0, j], Ya1[0, j], Ym1[0, j], Yr1[0, j], Aa1[0, j], Am1[0, j], cmu1[0, j], cms1[0, j], cau1[0, j], cas1[0, j], cu1[0, j], cs1[0, j], wu1[0, j], ws1[0, j], wr1[0, j], pa1[0, j], pm1[0, j], pr1[0, j]] = Output1
    [Output2, Rate2] = Calib(hdata2, Ndata2, Init2, Amgr2)
    [N2[0, j], Temp2[0, j], h2[0, j], Da2[0, j], Dm2[0, j], Dr2[0, j], H2[0, j], L2[0, j], Y2[0, j], YA2[0, j], Ya2[0, j], Ym2[0, j], Yr2[0, j], Aa2[0, j], Am2[0, j], cmu2[0, j], cms2[0, j], cau2[0, j], cas2[0, j], cu2[0, j], cs2[0, j], wu2[0, j], ws2[0, j], wr2[0, j], pa2[0, j], pm2[0, j], pr2[0, j]] = Output2
    [Aag2, Amg2, tu2, ts2] = Rate2
    
    MigCostU1 = tu1 * mcU1                                      # Migration cost for unskilled labor in country 1
    MigCostS1 = ts1 * mcS1                                      # Migration cost for skilled labor in country 1
    MigCostU2 = tu2 * mcU2                                      # Migration cost for unskilled labor in country 2
    MigCostS2 = ts2 * mcS2                                      # Migration cost for skilled labor in country 2
    
    wrs[0, j] = ws2[0, j]/ws1[0, j]
    wru[0, j] = wu2[0, j]/wu1[0, j]
    
    for i in range(T - 1):
        Con = RCP[j][i + 1] - RCP[j][0]
        
        Temp1[i + 1, j] = Temp1[0, j] + mu1 * Con**mu2 * (1 - mu3 * Temp1[0, j])
        Da1[i + 1, j] = max(0.001, g0a + g1a * Temp1[i + 1, j] + g2a * Temp1[i + 1, j]**2)
        Dm1[i + 1, j] = max(0.001, g0m + g1m * Temp1[i + 1, j] + g2m * Temp1[i + 1, j]**2)
        Dr1[i + 1, j] = Dm1[i + 1, j]/Da1[i + 1, j]

        Temp2[i + 1, j] = Temp2[0, j] + mu1 * Con**mu2 * (1 - mu3 * Temp2[0, j])
        Da2[i + 1, j] = max(0.001, g0a + g1a * Temp2[i + 1, j] + g2a * Temp2[i + 1, j]**2)
        Dm2[i + 1, j] = max(0.001, g0m + g1m * Temp2[i + 1, j] + g2m * Temp2[i + 1, j]**2)
        Dr2[i + 1, j] = Dm2[i + 1, j]/Da2[i + 1, j]
        
        Aa1[i + 1, j] = Aa1[i, j] * (1 + Aag1)
        Am1[i + 1, j] = Am1[i, j] * (1 + Amg1)
        Ar1[i + 1, j] = Am1[i + 1, j]/Aa1[i + 1, j]
        
        Aa2[i + 1, j] = Aa2[i, j] * (1 + Aag2)
        Am2[i + 1, j] = Am2[i, j] * (1 + Amg2)
        Ar2[i + 1, j] = Am2[i + 1, j]/Aa2[i + 1, j]
        
        def Mig(nx):
            # M1: country I
            # M2: country II
            [nuM1, nsM1, nuM2, nsM2] = nx
            
            HM1 = nsM1 * NM1 * (1 - bMs12) + nsM2 * NM2 * bMs21
            LM1 = nuM1 * NM1 * (1 - bMu12) + nuM2 * NM2 * bMu21
            hM1 = HM1/LM1
            YaM1 = AaM1 * LM1 * DaM1
            YmM1 = AmM1 * HM1 * DmM1
            wrM1 = np.exp(np.log((1 - alpha)/alpha) - 1/eps * np.log(hM1) - (1 - eps)/eps * np.log(DrM1 * ArM1))
            cmuM1 = YmM1 / (HM1 * wrM1 + LM1)
            cmsM1 = cmuM1 * wrM1
            cauM1 = YaM1 / (HM1 * wrM1 + LM1)
            casM1 = cauM1 * wrM1    
            cuM1 = (alpha * cauM1**((eps - 1)/eps) + (1 - alpha) * cmuM1**((eps - 1)/eps))**(eps/(eps - 1))
            csM1 = (alpha * casM1**((eps - 1)/eps) + (1 - alpha) * cmsM1**((eps - 1)/eps))**(eps/(eps - 1))
            wuM1 = cuM1 / (1 - gamma)
            wsM1 = csM1 / (1 - gamma)
            
            HM2 = nsM2 * NM2 * (1 - bMs21) + nsM1 * NM1 * bMs12
            LM2 = nuM2 * NM2 * (1 - bMu21) + nuM1 * NM1 * bMu12
            hM2 = HM2/LM2
            YaM2 = AaM2 * LM2 * DaM2
            YmM2 = AmM2 * HM2 * DmM2
            wrM2 = np.exp(np.log((1 - alpha)/alpha) - 1/eps * np.log(hM2) - (1 - eps)/eps * np.log(DrM2 * ArM2))            
            cmuM2 = YmM2 / (HM2 * wrM2 + LM2)
            cmsM2 = cmuM2 * wrM2
            cauM2 = YaM2 / (HM2 * wrM2 + LM2)
            casM2 = cauM2 * wrM2    
            cuM2 = (alpha * cauM2**((eps - 1)/eps) + (1 - alpha) * cmuM2**((eps - 1)/eps))**(eps/(eps - 1))
            csM2 = (alpha * casM2**((eps - 1)/eps) + (1 - alpha) * cmsM2**((eps - 1)/eps))**(eps/(eps - 1))
            wuM2 = cuM2 / (1 - gamma)
            wsM2 = csM2 / (1 - gamma)            

            gamma1 = nuM1 * (tuM1 * (1 + bMu12 * mcU1)) + nsM1 * (tsM1 * (1 + bMs12 * mcS1))
            dff11 = abs(gamma - gamma1)
            PwsM1 = bMs12 * (bMu12 * wsM2 / (nsM1 * wsM2 + nuM1 * wuM2) + (1 - bMu12) * wsM2 / (nsM1 * wsM2 + nuM1 * wuM1)) + (1 - bMs12) * (bMu12 *wsM1 / (nsM1 * wsM1 + nuM1 * wuM2) + (1 - bMu12) *wsM1 / (nsM1 * wsM1 + nuM1 * wuM1))
            PwuM1 = bMu12 * (bMs12 * wuM2 / (nsM1 * wsM2 + nuM1 * wuM2) + (1 - bMs12) * wuM2 / (nsM1 * wsM1 + nuM1 * wuM2)) + (1 - bMu12) * (bMs12 *wuM1 / (nsM1 * wsM2 + nuM1 * wuM1) + (1 - bMs12) *wuM1 / (nsM1 * wsM1 + nuM1 * wuM1))
            dff12 = abs(PwsM1/PwuM1 - (tsM1 * (1 + bMs12 * mcS1))/(tuM1 * (1 + bMu12 * mcU1)))
            
            gamma2 = nuM2 * (tuM2 * (1 + bMu21 * mcU2)) + nsM2 * (tsM2 * (1 + bMs21 * mcS2))
            dff21 = abs(gamma - gamma2)
            PwsM2 = bMs21 * (bMu21 * wsM1 / (nsM2 * wsM1 + nuM2 * wuM1) + (1 - bMu21) * wsM1 / (nsM2 * wsM1 + nuM2 * wuM2)) + (1 - bMs21) * (bMu21 *wsM2 / (nsM2 * wsM2 + nuM2 * wuM1) + (1 - bMu21) *wsM2 / (nsM2 * wsM2 + nuM2 * wuM2))
            PwuM2 = bMu21 * (bMs21 * wuM1 / (nsM2 * wsM1 + nuM2 * wuM1) + (1 - bMs21) * wuM1 / (nsM2 * wsM2 + nuM2 * wuM1)) + (1 - bMu21) * (bMs21 *wuM2 / (nsM2 * wsM1 + nuM2 * wuM2) + (1 - bMs21) *wuM2 / (nsM2 * wsM2 + nuM2 * wuM2))
            dff22 = abs(PwsM2/PwuM2 - (tsM2 * (1 + bMs21 * mcS2))/(tuM2 * (1 + bMu21 * mcU2)))
            
#            return abs(dff11 + dff12 + dff21 + dff22)
            return(dff11, dff12, dff21, dff22)
       
        tsM1 = ts1
        tuM1 = tu1
        NM1 = N1[i, j]
        AaM1 = Aa1[i + 1, j]
        AmM1 = Am1[i + 1, j]
        DaM1 = Da1[i + 1, j]
        DmM1 = Dm1[i + 1, j]
        DrM1 = Dr1[i + 1, j]
        ArM1 = Ar1[i + 1, j]

        tsM2 = ts2
        tuM2 = tu2
        NM2 = N2[i, j]
        AaM2 = Aa2[i + 1, j]
        AmM2 = Am2[i + 1, j]
        DaM2 = Da2[i + 1, j]
        DmM2 = Dm2[i + 1, j]
        DrM2 = Dr2[i + 1, j]
        ArM2 = Ar2[i + 1, j]
        
#        bs12[i + 1, j] = max(0, (1 - np.sum(ws1[:, j])/np.sum(ws2[:, j]))/pscl)
#        bu12[i + 1, j] = max(0, (1 - np.sum(wu1[:, j])/np.sum(wu2[:, j]))/pscl)
#        bs21[i + 1, j] = max(0, (1 - np.sum(ws2[:, j])/np.sum(ws1[:, j]))/pscl)
#        bu21[i + 1, j] = max(0, (1 - np.sum(wu2[:, j])/np.sum(wu1[:, j]))/pscl)
       
#        bs12[i + 1, j] = max(0, (1 - Dm1[i, j]/Dm2[i, j])/pscl)
#        bu12[i + 1, j] = max(0, (1 - Da1[i, j]/Da2[i, j])/pscl)
#        bs21[i + 1, j] = max(0, (1 - Dm2[i, j]/Dm1[i, j])/pscl)
#        bu21[i + 1, j] = max(0, (1 - Da2[i, j]/Da1[i, j])/pscl)

        # Climate-dependent Migration
        bs12[i + 1, j] = np.exp(-14.797 + 3.755 * np.log(Temp1[i, j])) * (1 - mcS1) * (wrs[i, j]/wrs[0, j])
        bu12[i + 1, j] = np.exp(47.163 - 16.212 * np.log(Temp1[i, j])) * (1 - mcU1) * (wru[i, j]/wru[0, j])
        bs21[i + 1, j] = 0
        bu21[i + 1, j] = 0

        # Climate-independent Migration
#        bs12[i + 1, j] = np.exp(-14.797 + 3.755 * np.log(Temp1[0, j])) * (1 - mcS1) * (wrs[0, j]/wrs[0, j])
#        bu12[i + 1, j] = np.exp(47.163 - 16.212 * np.log(Temp1[0, j])) * (1 - mcU1) * (wru[0 , j]/wru[0, j])
#        bs21[i + 1, j] = 0
#        bu21[i + 1, j] = 0

        # No Migration
#        bs12[i + 1, j] = 0
#        bu12[i + 1, j] = 0
#        bs21[i + 1, j] = 0
#        bu21[i + 1, j] = 0
        
        bMs12 = bs12[i + 1, j]
        bMu12 = bu12[i + 1, j]
        bMs21 = bs21[i + 1, j]
        bMu21 = bu21[i + 1, j]
        
        x0 = [0.1, 0.1, 0.1, 0.1]
        
#        bnds = [(0.001, gamma/tu1), (0.001, gamma/ts1), (0.001, gamma/tu2), (0.001, gamma/ts2)]
#        ftol = 1e-12
#        epsil = 1e-6
#        maxiter = 10000
#        res = minimize(Mig, x0, method='SLSQP', bounds=bnds, options={'ftol': ftol, 'eps': epsil, 'disp': True, 'maxiter': maxiter})
#        nsol = res.x
        
#        res = fsolve(Mig, x0)
#        nsol = res
        res = root(Mig, x0, method='lm')
        nsol = res.x
        
#        print(np.around(Mig(nsol)))
        
        nu1[i, j] = nsol[0]
        ns1[i, j] = nsol[1]
        nu2[i, j] = nsol[2]
        ns2[i, j] = nsol[3] 

        hn1[i, j] = ns1[i, j]/nu1[i, j]
        hn2[i, j] = ns2[i, j]/nu2[i, j] 
        
        Ms12[i + 1, j] = bs12[i + 1, j] * N1[i, j] * ns1[i, j]
        Mu12[i + 1, j] = bu12[i + 1, j] * N1[i, j] * nu1[i, j]
        Ms21[i + 1, j] = bs21[i + 1, j] * N2[i, j] * ns2[i, j]
        Mu21[i + 1, j] = bu21[i + 1, j] * N2[i, j] * nu2[i, j]
        
        H1[i + 1, j] = ns1[i, j] * N1[i, j] * (1 - bs12[i + 1, j]) + bs21[i + 1, j] * ns2[i, j] * N2[i, j]
        L1[i + 1, j] = nu1[i, j] * N1[i, j] * (1 - bu12[i + 1, j]) + bu21[i + 1, j] * nu2[i, j] * N2[i, j]
        h1[i + 1, j] = H1[i + 1, j]/L1[i + 1, j]
        wr1[i + 1, j] = np.exp(np.log((1 - alpha)/alpha) - 1/eps * np.log(h1[i + 1, j]) - (1 - eps)/eps * np.log(DrM1 * ArM1))
        N1[i + 1, j] = L1[i + 1, j] + H1[i + 1, j]
        Ya1[i + 1, j] = Aa1[i + 1, j] * L1[i + 1, j] * Da1[i + 1, j]
        Ym1[i + 1, j] = Am1[i + 1, j] * H1[i + 1, j] * Dm1[i + 1, j]
        Yr1[i + 1, j] = Ym1[i + 1, j] / Ya1[i + 1, j]
        pr1[i + 1, j] = (Yr1[i + 1, j])**(-1/eps) * alpha / (1 - alpha)
        ca = Ya1[i + 1, j] / N1[i + 1, j]
        cmu1[i + 1, j] = Ym1[i + 1, j] / (H1[i + 1, j] * wr1[i + 1, j] + L1[i + 1, j])
        cms1[i + 1, j] = cmu1[i + 1, j] * wr1[i + 1, j]
        cau1[i + 1, j] = Ya1[i + 1, j] / (H1[i + 1, j] * wr1[i + 1, j] + L1[i + 1, j])
        cas1[i + 1, j] = cau1[i + 1, j] * wr1[i + 1, j]    
        cu1[i + 1, j] = (alpha * cau1[i + 1, j]**((eps - 1)/eps) + (1 - alpha) * cmu1[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        cs1[i + 1, j] = (alpha * cas1[i + 1, j]**((eps - 1)/eps) + (1 - alpha) * cms1[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        wu1[i + 1, j] = cu1[i + 1, j] / (1 - gamma)
        ws1[i + 1, j] = cs1[i + 1, j] / (1 - gamma)
        pa1[i + 1, j] = wu1[i + 1, j] / (Da1[i + 1, j] * Aa1[i + 1, j])
        pm1[i + 1] = ws1[i + 1, j] / (Dm1[i + 1, j] * Am1[i + 1, j])    
        Y1[i + 1, j] = (alpha * Ya1[i + 1, j]**((eps -1)/eps) + (1 - alpha) * Ym1[i + 1, j]**((eps -1)/eps))**(eps/(eps - 1))
        YA1[i + 1, j] = Y1[i + 1, j] / N1[i + 1, j]
        Pop1[i, j] = N1[i + 1, j] + N1[i, j]
        Yp1[i, j] = Y1[i, j] / Pop1[i, j] * 10**6
                
        H2[i + 1, j] = ns2[i, j] * N2[i, j] * (1 - bs21[i + 1, j]) + bs12[i + 1, j] * ns1[i, j] * N1[i, j]
        L2[i + 1, j] = nu2[i, j] * N2[i, j] * (1 - bu21[i + 1, j]) + bu12[i + 1, j] * nu1[i, j] * N1[i, j]
        h2[i + 1, j] = H2[i + 1, j]/L2[i + 1, j]
        wr2[i + 1, j] = np.exp(np.log((1 - alpha)/alpha) - 1/eps * np.log(h2[i + 1, j]) - (1 - eps)/eps * np.log(DrM2 * ArM2))
        N2[i + 1, j] = L2[i + 1, j] + H2[i + 1, j]
        Ya2[i + 1, j] = Aa2[i + 1, j] * L2[i + 1, j] * Da2[i + 1, j]
        Ym2[i + 1, j] = Am2[i + 1, j] * H2[i + 1, j] * Dm2[i + 1, j]
        Yr2[i + 1, j] = Ym2[i + 1, j] / Ya2[i + 1, j]
        pr2[i + 1, j] = (Yr2[i + 1, j])**(-1/eps) * alpha / (1 - alpha)
        ca = Ya2[i + 1, j] / N2[i + 1, j]
        cmu2[i + 1, j] = Ym2[i + 1, j] / (H2[i + 1, j] * wr2[i + 1, j] + L2[i + 1, j])
        cms2[i + 1, j] = cmu2[i + 1, j] * wr2[i + 1, j]
        cau2[i + 1, j] = Ya2[i + 1, j] / (H2[i + 1, j] * wr2[i + 1, j] + L2[i + 1, j])
        cas2[i + 1, j] = cau2[i + 1, j] * wr2[i + 1, j]    
        cu2[i + 1, j] = (alpha * cau2[i + 1, j]**((eps - 1)/eps) + (1 - alpha) * cmu2[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        cs2[i + 1, j] = (alpha * cas2[i + 1, j]**((eps - 1)/eps) + (1 - alpha) * cms2[i + 1, j]**((eps - 1)/eps))**(eps/(eps - 1))
        wu2[i + 1, j] = cu2[i + 1, j] / (1 - gamma)
        ws2[i + 1, j] = cs2[i + 1, j] / (1 - gamma)
        pa2[i + 1, j] = wu2[i + 1, j] / (Da2[i + 1, j] * Aa2[i + 1, j])
        pm2[i + 1] = ws2[i + 1, j] / (Dm2[i + 1, j] * Am2[i + 1, j])    
        Y2[i + 1, j] = (alpha * Ya2[i + 1, j]**((eps -1)/eps) + (1 - alpha) * Ym2[i + 1, j]**((eps -1)/eps))**(eps/(eps - 1))
        YA2[i + 1, j] = Y2[i + 1, j] / N2[i + 1, j]
        Pop2[i, j] = N2[i + 1, j] + N2[i, j]
        Yp2[i, j] = Y2[i, j] / Pop2[i, j] * 10**6

        wrs[i + 1, j] = ws2[i + 1, j]/ws1[i + 1, j]
        wru[i + 1, j] = wu2[i + 1, j]/wu1[i + 1, j]
 
        wsM1 = ws1[i + 1, j]
        wuM1 = wu1[i + 1, j]
        wsM2 = ws2[i + 1, j]
        wuM2 = wu2[i + 1, j]
       
        nuM1 = nu1[i, j]
        nsM1 = ns1[i, j]
        nuM2 = nu2[i, j]
        nsM2 = ns2[i, j]

        PwsM1[i + 1, j] = bMs12 * (bMu12 * wsM2 / (nsM1 * wsM2 + nuM1 * wuM2) + (1 - bMu12) * wsM2 / (nsM1 * wsM2 + nuM1 * wuM1)) + (1 - bMs12) * (bMu12 *wsM1 / (nsM1 * wsM1 + nuM1 * wuM2) + (1 - bMu12) *wsM1 / (nsM1 * wsM1 + nuM1 * wuM1))
        PwuM1[i + 1, j] = bMu12 * (bMs12 * wuM2 / (nsM1 * wsM2 + nuM1 * wuM2) + (1 - bMs12) * wuM2 / (nsM1 * wsM1 + nuM1 * wuM2)) + (1 - bMu12) * (bMs12 *wuM1 / (nsM1 * wsM2 + nuM1 * wuM1) + (1 - bMs12) *wuM1 / (nsM1 * wsM1 + nuM1 * wuM1))        
        dffx1[i + 1, j] = PwsM1[i + 1, j]/PwuM1[i + 1, j]
        dffx11[i + 1, j] = (tsM1 * (1 + bMs12 * mcS1))/(tuM1 * (1 + bMu12 * mcU1))
        
        PwsM2[i + 1, j] = bMs21 * (bMu21 * wsM1 / (nsM2 * wsM1 + nuM2 * wuM1) + (1 - bMu21) * wsM1 / (nsM2 * wsM1 + nuM2 * wuM2)) + (1 - bMs21) * (bMu21 *wsM2 / (nsM2 * wsM2 + nuM2 * wuM1) + (1 - bMu21) *wsM2 / (nsM2 * wsM2 + nuM2 * wuM2))
        PwuM2[i + 1, j] = bMu21 * (bMs21 * wuM1 / (nsM2 * wsM1 + nuM2 * wuM1) + (1 - bMs21) * wuM1 / (nsM2 * wsM2 + nuM2 * wuM1)) + (1 - bMu21) * (bMs21 *wuM2 / (nsM2 * wsM1 + nuM2 * wuM2) + (1 - bMs21) *wuM2 / (nsM2 * wsM2 + nuM2 * wuM2))
        dffx2[i + 1, j] = PwsM2[i + 1, j]/PwuM2[i + 1, j]
        dffx22[i + 1, j] = (tsM2 * (1 + bMs21 * mcS2))/(tuM2 * (1 + bMu21 * mcU2))
        
# ===================================================== Output ===================================================== #    
x = [2000, 2020, 2040, 2060, 2080, 2100]
xend = 2120

plt.plot(x, Ndata1, 'r:', label = "Data")
plt.plot(x, N1[:, 0], 'r', label = "Baseline")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Adult population in country 1 (Baseline case)')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Ndata2, 'b:', label = "Data")
plt.plot(x, N2[:, 0], 'blue', label = "Baseline")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Adult population in country 2 (Baseline case)')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, N1[:, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, N1[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, N1[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, N1[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Population of country 1')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, N2[:, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, N2[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, N2[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, N2[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Population of country 2')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Ms12[:, 0] * scl, 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Ms12[:, 1] * scl, 'green', label = "RCP 4.5")
plt.plot(x, Ms12[:, 2] * scl, 'cyan', label = "RCP 6.0")
plt.plot(x, Ms12[:, 3] * scl, 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Migration of skilled labor from country 1 to 2')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(ymax = (np.amax(Ms12) + 1/pscl) * scl, ymin = 0)
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Ms21[:, 0] * scl, 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Ms21[:, 1] * scl, 'green', label = "RCP 4.5")
plt.plot(x, Ms21[:, 2] * scl, 'cyan', label = "RCP 6.0")
plt.plot(x, Ms21[:, 3] * scl, 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Migration of skilled labor from country 2 to 1')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(ymax = (np.amax(Ms21) + 1/pscl) * scl, ymin = 0)
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Mu12[:, 0] * scl, 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Mu12[:, 1] * scl, 'green', label = "RCP 4.5")
plt.plot(x, Mu12[:, 2] * scl, 'cyan', label = "RCP 6.0")
plt.plot(x, Mu12[:, 3] * scl, 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Migration of unskilled labor from country 1 to 2')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(ymax = (np.amax(Mu12) + 1/pscl) * scl, ymin = 0)
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Mu21[:, 0] * scl, 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Mu21[:, 1] * scl, 'green', label = "RCP 4.5")
plt.plot(x, Mu21[:, 2] * scl, 'cyan', label = "RCP 6.0")
plt.plot(x, Mu21[:, 3] * scl, 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('millions')
plt.title('Migration of unskilled labor from country 2 to 1')
axes = plt.gca()
#axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(ymax = (np.amax(Mu21) + 1/pscl) * scl, ymin = 0)
plt.xticks(np.arange(min(x), max(x), 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hdata1, 'k:', label = "Data")
plt.plot(x, h1[:, 0], 'blue', label = "Baseline: RCP 2.6")
plt.plot(x, h1[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, h1[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, h1[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Ratio of skilled to unskilled adults in country 1')
axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, hdata2, 'k:', label = "Data")
plt.plot(x, h2[:, 0], 'blue', label = "Baseline: RCP 2.6")
plt.plot(x, h2[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, h2[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, h2[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Ratio of skilled to unskilled adults in country 2')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Dr1[:, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Dr1[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, Dr1[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, Dr1[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Ratio of Manufacturing to Agricultural Damages in country 1')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x, Dr2[:, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x, Dr2[:, 1], 'green', label = "RCP 4.5")
plt.plot(x, Dr2[:, 2], 'cyan', label = "RCP 6.0")
plt.plot(x, Dr2[:, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Ratio of Manufacturing to Agricultural Damages in country 2')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x[0:T - 1], Yp1[0:T - 1, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x[0:T - 1], Yp1[0:T - 1, 1], 'green', label = "RCP 4.5")
plt.plot(x[0:T - 1], Yp1[0:T - 1, 2], 'cyan', label = "RCP 6.0")
plt.plot(x[0:T - 1], Yp1[0:T - 1, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('1990 GK$/person')
plt.title('Output per capita in country 1')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend - 20, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x[0:T - 1], Yp2[0:T - 1, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x[0:T - 1], Yp2[0:T - 1, 1], 'green', label = "RCP 4.5")
plt.plot(x[0:T - 1], Yp2[0:T - 1, 2], 'cyan', label = "RCP 6.0")
plt.plot(x[0:T - 1], Yp2[0:T - 1, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('1990 GK$/person')
plt.title('Output per capita in country 2')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend - 20, 20))
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x[0:T - 1], wr1[0:T - 1, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x[0:T - 1], wr1[0:T - 1, 1], 'green', label = "RCP 4.5")
plt.plot(x[0:T - 1], wr1[0:T - 1, 2], 'cyan', label = "RCP 6.0")
plt.plot(x[0:T - 1], wr1[0:T - 1, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Skilled to unskilled wage ratio in country 1')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend - 20, 20))
plt.ylim(ymax = np.amax(wr1)+1, ymin = 0)
plt.legend(loc=2, prop={'size':8})
plt.show()

plt.plot(x[0:T - 1], wr2[0:T - 1, 0], 'b--', label = "Baseline: RCP 2.6")
plt.plot(x[0:T - 1], wr2[0:T - 1, 1], 'green', label = "RCP 4.5")
plt.plot(x[0:T - 1], wr2[0:T - 1, 2], 'cyan', label = "RCP 6.0")
plt.plot(x[0:T - 1], wr2[0:T - 1, 3], 'orange', label = "RCP 8.5")
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('Skilled to unskilled wage ratio in country 2')
axes = plt.gca()
plt.xticks(np.arange(min(x), xend - 20, 20))
plt.ylim(ymax = np.amax(wr2)+1, ymin = 0)
plt.legend(loc=2, prop={'size':8})
plt.show()

#plt.plot(x[0:T - 1], YP[0:T - 1], 'darkgreen')
#plt.xlabel('Time')
#plt.ylabel('1000 GK$ per person')
#plt.title('GDP per capita')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), max(x), 20))
#plt.show()

#plt.plot(x, Dr, 'brown')
#plt.xlabel('Time')
#plt.ylabel('Ratio')
#plt.title('Damage ratio (manufacturing to agricultural)')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()
#
#plt.plot(x[0:T - 1], nu[0:T - 1], 'c')
#plt.xlabel('Time')
#plt.ylabel('Population')
#plt.title('Number of unskilled children per parent')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), max(x), 20))
#plt.show()
#
#plt.plot(x[0:T - 1], ns[0:T - 1], 'g')
#plt.xlabel('Time')
#plt.ylabel('Population')
#plt.title('Number of skilled children per parent')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), max(x), 20))
#plt.show()
#
#plt.plot(x, L, 'r')
#plt.xlabel('Time')
#plt.ylabel('Population (millions)')
#plt.title('Unskilled labor')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()
#
#plt.plot(x, H, 'y')
#plt.xlabel('Time')
#plt.ylabel('Population (millions)')
#plt.title('Skilled labor')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()
#
#plt.plot(x, pr, 'brown')
#plt.xlabel('Time')
#plt.ylabel('Ratio')
#plt.title('Price ratio (manufacturing to agricultural)')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()
#
#plt.plot(x, Y, 'darkcyan')
#plt.xlabel('Time')
#plt.ylabel('GK$')
#plt.title('GDP (1990 billion GK$)')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()
#
#plt.plot(x, Ar, 'chocolate')
#plt.xlabel('Time')
#plt.ylabel('level')
#plt.title('Technology ratio (Manufacturing to Agriculture)')
#axes = plt.gca()
#plt.xticks(np.arange(min(x), xend, 20))
#plt.show()

# =============================================== Export into Excel =============================================== #

def output(filename, sheet, list1, list2, list3, v):
    book = xlwt.Workbook(filename)
    sh = book.add_worksheet(sheet)

    v1_desc = 'alpha'
    v2_desc = 'eps'
    v3_desc = 'latitude1'
    v4_desc = 'latitude2'
    v5_desc = 'Tu1'
    v6_desc = 'Ts1'
    v7_desc = 'Tu2'
    v8_desc = 'Ts2'
    v9_desc = 'gama'
    v10_desc = 'mig cost unskilled 1 to 2'
    v11_desc = 'mig cost skilled 1 to 2'
    v12_desc = 'mig cost unskilled 2 to 1'
    v13_desc = 'mig cost skilled 2 to 1'
    desc = [v1_desc, v2_desc, v3_desc, v4_desc, v5_desc, v6_desc, v7_desc, v8_desc, v9_desc, v10_desc, v11_desc, v12_desc, v13_desc]

    row0_name = 'Time'
    
    row1_name = 'Nu1'
    row2_name = 'Ns1'
    row3_name = 'L1'
    row4_name = 'H1'  
    row5_name = 'N1'
    row6_name = 'Pop1'
    row7_name = 'h1'
    row8_name = 'wu1'
    row9_name = 'ws1'
    row10_name = 'Aa1'
    row11_name = 'Am1'
    row12_name = 'Da1'
    row13_name = 'Dm1'
    row14_name = 'Ya1'
    row15_name = 'Ym1'
    row16_name = 'Pa1'
    row17_name = 'Pm1'
    row18_name = 'hn1'
    row19_name = 'Yp1'
    
    row21_name = 'Nu2'
    row22_name = 'Ns2'
    row23_name = 'L2'
    row24_name = 'H2'  
    row25_name = 'N2'
    row26_name = 'Pop2'
    row27_name = 'h2'
    row28_name = 'wu2'
    row29_name = 'ws2'
    row30_name = 'Aa2'
    row31_name = 'Am2'
    row32_name = 'Da2'
    row33_name = 'Dm2'
    row34_name = 'Ya2'
    row35_name = 'Ym2'
    row36_name = 'Pa2'
    row37_name = 'Pm2'
    row38_name = 'hn2'
    row39_name = 'Yp2'
    
    row41_name = 'bu12'
    row42_name = 'bs12'
    row43_name = 'bu21'
    row44_name = 'bs21'
    
    n = 0
    for v_desc, v_v in zip(desc, v):
        sh.write(n, 0, v_desc)
        sh.write(n, 1, v_v)
        n = n + 1
    
    sh.write(n, 0, row0_name)
    
    sh.write(n + 1, 0, row1_name)
    sh.write(n + 2, 0, row2_name)
    sh.write(n + 3, 0, row3_name)
    sh.write(n + 4, 0, row4_name)
    sh.write(n + 5, 0, row5_name)
    sh.write(n + 6, 0, row6_name)
    sh.write(n + 7, 0, row7_name)
    sh.write(n + 8, 0, row8_name)
    sh.write(n + 9, 0, row9_name)
    sh.write(n + 10, 0, row10_name)
    sh.write(n + 11, 0, row11_name)
    sh.write(n + 12, 0, row12_name)
    sh.write(n + 13, 0, row13_name)
    sh.write(n + 14, 0, row14_name)
    sh.write(n + 15, 0, row15_name)
    sh.write(n + 16, 0, row16_name)
    sh.write(n + 17, 0, row17_name)
    sh.write(n + 18, 0, row18_name)
    sh.write(n + 19, 0, row19_name)

    sh.write(n + 21, 0, row21_name)
    sh.write(n + 22, 0, row22_name)
    sh.write(n + 23, 0, row23_name)
    sh.write(n + 24, 0, row24_name)
    sh.write(n + 25, 0, row25_name)
    sh.write(n + 26, 0, row26_name)
    sh.write(n + 27, 0, row27_name)
    sh.write(n + 28, 0, row28_name)
    sh.write(n + 29, 0, row29_name)
    sh.write(n + 30, 0, row30_name)
    sh.write(n + 31, 0, row31_name)
    sh.write(n + 32, 0, row32_name)
    sh.write(n + 33, 0, row33_name)
    sh.write(n + 34, 0, row34_name)
    sh.write(n + 35, 0, row35_name)
    sh.write(n + 36, 0, row36_name)
    sh.write(n + 37, 0, row37_name)
    sh.write(n + 38, 0, row38_name)
    sh.write(n + 39, 0, row39_name)
    
    sh.write(n + 41, 0, row41_name)
    sh.write(n + 42, 0, row42_name)
    sh.write(n + 43, 0, row43_name)
    sh.write(n + 44, 0, row44_name)
    
    for j in range(4):
        for indx , m in enumerate(range(2000, 2120, 20), 1):
            sh.write(n + 0, j * 10 + indx, m)
        for indx , m in enumerate(range(2000, 2120, 20), 1):
            sh.write(n + 20, j * 10 + indx, m)
        for indx , m in enumerate(range(2000, 2120, 20), 1):
            sh.write(n + 40, j * 10 + indx, m)
        for k in range(19):
            for indx , m in enumerate(list1[k][:, j], 1):
                sh.write(n + k + 1, j * 10 + indx, m) 
            for indx , m in enumerate(list2[k][:, j], 1):
                sh.write(n + k + 21, j * 10 + indx, m)
        for k in range(4):
            for indx , m in enumerate(list3[k][:, j], 1):
                sh.write(n + k + 41, j * 10 + indx, m)
    book.close()
    
output1 = [nu1, ns1, L1, H1, N1, Pop1, h1, wu1, ws1, Aa1, Am1, Da1, Dm1, Ya1, Ym1, pa1, pm1, hn1, Yp1]
output2 = [nu2, ns2, L2, H2, N2, Pop2, h2, wu2, ws2, Aa2, Am2, Da2, Dm2, Ya2, Ym2, pa2, pm2, hn2, Yp2]
output3 = [bu12, bs12, bu21, bs21]
par = [alpha, eps, latid1, latid2, tu1, ts1, tu2, ts2, gamma, mcU1, mcS1, mcU2, mcS2]
output('Res1.xlsx', 'Sheet1', output1, output2, output3, par)