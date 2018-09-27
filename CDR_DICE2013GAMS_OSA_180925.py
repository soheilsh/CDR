"""

"""
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# =========================================== Global Parameters =========================================== #
#%reset
global deltaT_true, nuG_true, Coef, ActA0, ActG0, ActS0, ActA0_ADP, ActG0_ADP
global al, sigma, etree, l, cost1, rr, forcoth
global Vh, Vb, App, m, miter, t, nuG_belief, deltaT_belief
global Mat_ADP, Mup_ADP, Mlo_ADP, Tat_ADP, Tlo_ADP, RFC_ADP, Geo_ADP
global Mat, Mup, Mlo, Tat, Tlo, RFC

# =========================================== Parameters =========================================== #
tstep = 5                                   #Years per Period
hzn = 1
rng_ADP = int(20/tstep + hzn)                                     
T = int(300/tstep)                              #Modeling time horizon
T0 = T - 10                                 #Graph time horizon

## Preferences  ##
elasmu = 1.45                               #Elasticity of marginal utility of consumption
prstp = 0.015                               #Initial rate of social time preference per year

## == Population and technology == ##
alpha = 0.3                                 #Capital elasticity in production function 
pop0 = 6838.00                              #Initial world population (millions)
popadj = 0.134*tstep/5                      #Growth rate to calibrate to 2050 pop projection
popasym = 10500.00                          #Asymptotic population (millions)
dk = 0.10                                   #Depreciation rate on capital (per year)
q0 = 63.69                                  #Initial world gross output (trill 2005 USD)
k0 = 135.00                                 #Initial capital value (trill 2005 USD)
a0 = 3.80                                   #Initial level of total factor productivity      
ga0 = 0.0790                                #Initial growth rate for TFP per 5 years 
dela = 0.0060                               #Decline rate of TFP per 5 years

## == Emissions parameters == ##
gsigma1 = -0.010                            #Initial growth of sigma (per year)
dsig = -0.0010                              #Decline rate of decarbonization (per period)
eland0 = 3.3                                #Carbon emissions from land 2010 (GtCO2 per year)
deland = 0.2                                #Decline rate of land emissions (per period)
e0 = 33.61                                  #Industrial emissions 2010 (GtCO2 per year)
miu0 = 0.039                                 #Initial emissions control rate for base case 2010 

## == Carbon cycle == ##
# == Initial Conditions == #
mat0 = 830.4                                #Initial Concentration in atmosphere 2010 (GtC)
mu0 = 1527.00                               #Initial Concentration in upper strata 2010 (GtC)
ml0 = 10010.00                              #Initial Concentration in lower strata 2010 (GtC)
mateq = 588.00                              #Equilibrium concentration atmosphere  (GtC)
mueq = 1350.00                              #Equilibrium concentration in upper strata (GtC)
mleq = 10000.00                             #Equilibrium concentration in lower strata (GtC)

# == Flow paramaters == #
b12 = 0.088*tstep/5                         #atmosphere to biosphere/shallow oceans (b12)
b11 = 1 - b12                               #atmosphere to atmosphere (b11)
b21 = b12 * mateq/ mueq                     #biosphere/shallow oceans to atmosphere (b21)
b23 = 0.0025*tstep/5                        #biosphere/shallow oceans to deep oceans (b23)
b22 = 1 - b21 - b23                         #biosphere/shallow oceans to biosphere/shallow oceans (b22)
b32 = b23 * mueq/ mleq                      #deep oceans to biosphere/shallow oceans (b32)
b33 = 1 - b32                               #deep oceans to deep oceans (b33) 
sig0 = e0/(q0*(1 - miu0))                   #Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)
beta0 = b12*b23

## == Climate model parameters == ##
t2xco2 = 2.9                                #Equilibrium temp impact (oC per doubling CO2)
fex0 = 0.25                                 #2010 forcings of non-CO2 GHG (Wm-2) 
fex1 = 0.70                                 #2100 forcings of non-CO2 GHG (Wm-2) 
tocean0 = 0.0068                            #Initial lower stratum temp change (C from 1900)
tatm0 = 0.80                                #Initial atmospheric temp change (C from 1900)

c10 = 0.098                                 #Initial climate equation coefficient for upper level
c1beta = 0.01243                            #Regression slope coefficient(SoA~Equil TSC) 

c1 = 0.098                                  #Climate equation coefficient for upper level
c3 = 0.088                                  #Transfer coefficient upper to lower stratum
c4 = 0.025                                  #CTransfer coefficient for lower level
fco22x = 3.8                                #Forcings of equilibrium CO2 doubling (Wm-2)
c1 =  c10 + c1beta*(t2xco2 - 2.9)

## == Climate damage parameters == ##
a10 = 0                                     #Initial damage intercept
a20 = 0.00267                               #Initial damage quadratic term
a1 = 0                                      #Damage intercept
a2 = 0.00267                                #Damage quadratic term
a3 = 2.00                                   #Damage exponent

## == Abatement cost == ##
expcost2 = 2.8                              #Exponent of control cost function
pback = 344                                 #Cost of backstop 2005$ per tCO2 2010
gback = 0.025                               #Initial cost decline backstop cost per period
limmiu = 1.2                                #Upper limit on control rate after 2150
tnopol = 45                                 #Period before which no emissions controls base
cprice0 = 1.0                               #Initial base carbon price (2005$ per tCO2)
gcprice = 0.02                              #Growth rate of base carbon price per year 

# == Other parameters == # 
lam = fco22x/ t2xco2                        #Climate model parameter
optlrsav = (dk + .004)/(dk + .004*elasmu + prstp)*alpha       #Optimal long-run savings rate used for transversality

# == CDR parameters == # 
Rexpcost2 = 2

#gamma0 = mueq * b23 / mleq
gamma0 = 0.5
gamma1 = [0.0002 * b for b in range(4)]
sz = len(gamma1)
#sz = 1
Rscale = 100
                      
# ============================================ State Variables ============================================ #

 # == Capital ($trill, 2005$) == #
K = k0 + np.zeros((sz, T))

 # == Atmospheric concentration of carbon (GTC) == #
Mat = mat0 + np.zeros((sz, T))

 # == Concentration in biosphere and upper oceans (GTC) == #
Mlo = ml0 + np.zeros((sz, T))

 # == Concentration in deep oceans (GTC) == #
Mup = mu0 + np.zeros((sz, T))

 # == CAtmospheric temperature (degrees Celsius above preindustrial) == #
Tat = tatm0 + np.zeros((sz, T))

 # == Lower ocean temperature (degrees Celsius above preindustrial) == #
Tlo = tocean0 + np.zeros((sz, T))

# ============================================ Exogenous Variables ============================================ #

# == Population == #
l = [0]*T                                   #Level of population and labor

# == Technology == #
al = [0]*T                                  #Level of total factor productivity
ga = [0]*T                                  #Growth rate of productivity

 # == Forcing == #
forcoth = [0]*(T+1)                         #Exogenous forcing for other greenhouse gases

# == Abetment Cost == #
theta1 = [0]*T                              #Adjusted cost for backstop
pbacktime = [0]*T                           #Backstop price
 
# == Emissions == #
sigma = [0]*T                               #Abatement cost function coefficient
gsig = [0]*T                                #Change in sigma (cumulative improvement of energy efficiency)
etree = [0]*T                               #Emissions from deforestation

# == Abatement == #
cost1 = [0]*T                               #Adjusted cost for backstop
costR1 = [0]*T                              #Adjusted cost for CDR

rr = [1]*T                                  #Average utility social discount rate

     # ============================================ Output Variables ============================================ #

Yt = np.zeros((sz, T))                              #Output gross of abatement cost and climate damage ($trill)
Qt = np.zeros((sz, T))                              #Output net of abatement cost and climate damage ($trill)
Et = np.zeros((sz, T))                              #Total carbon emissions (GTCO2 per year)
Eind = np.zeros((sz, T))                           #Industrial emissions (GTCO2 per year)
Eint = np.zeros((sz, T))                           #World emissions intensity (sigma)
Ft = np.zeros((sz, T))                             #Total increase in radiative forcing since preindustrial (Watts per square meter)
Dm = np.zeros((sz, T))                            #Total damage (fraction of gross output)
Dt = np.zeros((sz, T))                             #Climate damages (trillion $)
Bm = np.zeros((sz, T))                             #Abatement cost (fraction of output)
Bt = np.zeros((sz, T))                             #Abatement cost ($ trillion)
Rm = np.zeros((sz, T))                             #CDR cost (fraction of output)
Rt = np.zeros((sz, T))                             #CDR cost ($ trillion)
Sv = np.zeros((sz, T))                             #Saving ($trill, 2005$)
Cn = np.zeros((sz, T))                             #Consumption ($trill per year)
cn = np.zeros((sz, T))                             #Consumption per capita ($thous per year)
Ut = np.zeros((sz, T))                             #Total period utility
ut = np.zeros((sz, T))                             #Utility of p. c. consumption
b32 = np.zeros((sz, T))                            #lower ocean to upper ocean coeff
CDR = np.zeros((sz, T))                            #CDR (GTCO2 per year)
Outgas = np.zeros((sz, T))                         #Outgassing from ocean to atmosphere (GTCO2 per year)
Oeffective = np.zeros((sz, T))                           #ocean sink (GTCO2 per year)
Cp = np.zeros((sz, T)) 

AoptSA = np.zeros((sz, T))                              #Optimal Abatement rate
RoptSA = np.zeros((sz, T))                              #Optimal CDR rate
     
# ========================================= Exogenous variables ========================================= #

for i in range(T):
    if i == 0:
        al[i] = a0
        pbacktime[i] = pback
        gsig[i] = gsigma1/(1 + dsig)**tstep
        sigma[i] = sig0
        etree[i] = eland0
        l[i] = pop0
    else:
        rr[i] = rr[i-1]/(1 + prstp)**tstep
        gsig[i] = gsig[i-1] * (1 + dsig)**tstep
        sigma[i] = sigma[i-1]*np.exp(gsig[i]*tstep)
        al[i] = al[i-1]/(1 - ga[i-1])**(tstep/5)
        pbacktime[i] = pbacktime[i-1]*(1 - gback)**(tstep/5)
        etree[i] = etree[i-1] * (1 - deland)**(tstep/5)
        l[i] = l[i-1]*(popasym/l[i-1])**popadj
    ga[i] = ga0 * np.exp(-dela * tstep * i)
    cost1[i] = pbacktime[i] * sigma[i]/expcost2/1000
    costR1[i] = pbacktime[i] * sigma[i]/expcost2/1000
    if i<(90/tstep):
        forcoth[i] = fex0 + (1/18)*(tstep/5) * (fex1 - fex0) * (tstep/5 + i)
    else:
        forcoth[i] = fex1
Ft[0] = fco22x * (np.log(mat0/mateq))/np.log(2) + forcoth[0]

# =================================== Truncated lognormal Distribution =================================== #

#    Truncated lognormal Distribution
#    Calculates the lognormal distribution for values between dt_l and dt_u

def lognorm_trunc( p, low, up, mu, sig  ):
    cdf2 = lognorm.cdf(up, sig, loc=0, scale=np.exp(mu))
    cdf1 = lognorm.cdf(low, sig, loc=0, scale=np.exp(mu))
    pc = p * (cdf2 - cdf1) + cdf1
    dt = lognorm.ppf(pc, sig, loc=0, scale=np.exp(mu))
    return dt

# costR11 * ========================================= Transition Function ========================================= #

def state( ST1, EX1, Act1, Inf1 ):
    
    [K1, Matx1, Mlox1, Tatx1, Tlox1, RFCx1] = ST1
    [A1, sig1, Eland1, L1, cost11, costR11, df1, pb1, Fexx1, Fexx2] = EX1
    [xa1, xr01] = Act1
    gamma1x = Inf1
    xr1 = xr01 * Rscale
    Y1 = A1 * (L1/1000)**(1 - alpha) * K1**alpha
    
    Eind1 = sig1 * Y1 * (1 - xa1)
    
    E1 = Eind1 + Eland1
    
    Eint1 = E1/Y1
    
    Dm1 = (a1 * Tatx1 + a2 * Tatx1**a3)
    
    D1 = Y1 * Dm1
    
    Bm1 = cost11 * xa1**expcost2
    
    B1 = Bm1 * Y1
    
    Rm1 = (5 * cost11 * 1.0**expcost2 * q0 ) / (costR11 * (1.0 * sig0 * q0)**Rexpcost2)
    
    R1 = Rm1 * costR11 * xr1**Rexpcost2
    
    Q1 = Y1 - D1 - B1 - R1
    
    S1 = optlrsav * Q1
    
    Con1 = Q1 - S1
    
    con1 = (Con1/L1) * 1000
    
    u1 = (con1**(1 - elasmu) - 1)/ (1 - elasmu) - 1
    
    U1 = u1 * L1 * df1
    
    Cp1   = pb1 * (xa1)**(expcost2-1)
    
    sink1 = beta0 * Mlox1
    Oeffective1 = gamma0 - gamma1x * (Matx1 - mat0)
    Outgas1 = (Oeffective1) * xr1
    
    Matx2 = Matx1 + E1 * tstep/3.666 - sink1 + Outgas1 * tstep/3.666 - xr1 * tstep/3.666

    Mlox2 = Mlox1 + sink1 - Outgas1 * tstep/3.666

    RFCx2 = fco22x * (np.log(Matx2/mateq))/np.log(2) + Fexx2
    Tatx2 = Tatx1 + c1 * (RFCx2 - (fco22x/t2xco2) * Tatx1 - c3 * (Tatx1 - Tlox1))
    Tlox2 = Tlox1 + c4 * (Tatx1 - Tlox1)
  
    K2 = (1 - dk)**tstep * K1 + tstep * S1

    ST2 = [K2, Matx2, Mlox2, Tatx2, Tlox2, RFCx2]
    LEV2 = [Y1, Q1, E1, Eind1, Eint1, Dm1, D1, Bm1, B1, Rm1, R1, S1, Con1, con1, U1, u1, Cp1, Oeffective1, Outgas1]
    
    return ( ST2, LEV2 )

# =========================================== Welfare Function (DICE model) =========================================== #

def fDICE(v):  
    W = 0
    for i in range(T-1):
        
        STi = [K[j, i], Mat[j, i], Mlo[j, i], Tat[j, i], Tlo[j, i], Ft[j, i]]
        EXi = [al[i], sigma[i], etree[i], l[i], cost1[i], costR1[i], rr[i], pbacktime[i], forcoth[i], forcoth[i + 1]]
        Acti = [v[i], v[i + T]]
            
        ( STii, LEVii ) = state( STi, EXi, Acti, INFF )
            
        [K[j, i + 1] , Mat[j, i + 1], Mlo[j, i + 1], Tat[j, i + 1], Tlo[j, i + 1], Ft[j, i + 1]] = STii
        [Yt[j, i], Qt[j, i], Et[j, i], Eind[j, i], Eint[j, i], Dm[j, i], Dt[j, i], Bm[j, i], Bt[j, i], Rm[j, i], Rt[j, i], Sv[j, i], Cn[j, i], cn[j, i], Ut[j, i], ut[j, i], Cp[j, i], Oeffective[j, i], Outgas[j, i]] = LEVii
        
        W = W +  Ut[j, i] * tstep
    return -W
    
# ======================================== Optimization Algorithm (DICE model) ======================================== #

def DICEalg( gamma1x, j ):
    global INFF
    INFF = gamma1x
    x0 = 2 * T * [0.00]
    # == bounds == #
    bnds = 2 * T * [(0.0, 1)]
    bnds[0] = (0.039, 0.039)
    bnds[T:2 * T] = T * [(0.00, 1.00)]
    bnds[T] = (0.0, 0.0)
    
    # == optimization == #
    ftol = 1e-12
    eps = 1e-6
    maxiter = 10000
        
    res = minimize(fDICE, x0, method='SLSQP', bounds=bnds, options={'ftol': ftol, 'eps': eps, 'disp': True, 'maxiter': maxiter})
    resDICE = res.x
    AoptSA[j] = resDICE[0:T]
    RoptSA[j] = resDICE[T:2*T]
    return AoptSA, RoptSA

# ======================================== Comparative Analysis ======================================== #

for j in range(sz):
    gamma1j = gamma1[j]
    DICEalg( gamma1j, j )

T0 = T - 10
Xaxis = range(2010, 2010 + tstep * T0, tstep)

for j in range(sz):
    plt.plot(Xaxis[0:T0], AoptSA[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Rate')
plt.title('Optimal abatement rate')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Rscale*RoptSA[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Removed concentraion (GtCO2)')
plt.title('Optimal level of CDR')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Et[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Emissions (GtCO2)')
plt.title('Optimal level of Emissions')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()    
    
for j in range(sz):
    plt.plot(Xaxis[0:T0], Outgas[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level (GtCO2)')
plt.title('Outgasing from ocean to atmosphere')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Dt[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level')
plt.title('Damages')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Bt[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level')
plt.title('Abatement cost')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Rt[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level')
plt.title('CDR cost')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Yt[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level')
plt.title('Economic Output')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Tat[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level')
plt.title('Temperature')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Mat[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level (GtCO2)')
plt.title('Atmospheric Concentrations')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Mlo[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Level (GtCO2)')
plt.title('Ocean Concentrations')
plt.legend(loc=1, prop={'size':8})
#plt.ylim(ymax = 4, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()

for j in range(sz):
    plt.plot(Xaxis[0:T0], Oeffective[j, 0:T0], label = "gamma1 = " + str(gamma1[j]) )
plt.xlabel('Time (years)')
plt.ylabel('Rate')
plt.title('Outgassing Effectiveness')
plt.legend(loc=1, prop={'size':8})
plt.ylim(ymax = 1, ymin = 0)
plt.xlim(xmax = max(Xaxis), xmin = min(Xaxis))
plt.xticks(np.arange(min(Xaxis), max(Xaxis), 5 * tstep))
plt.show()