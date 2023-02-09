# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:40:17 2022

@author: micha
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fsolve, root

# 2 factor model parameters
x0 = -0.005
alpha = 3
sigma = 0.01
y0 = 0.005
beta = 1
eta = 0.005

tmonths = 12
tdays = 21

# phi parameters 
a = 0.02
b = 0.05
lam = 0.75

#%% Question 2

def phi(t):
    return a + b*(((1-np.exp(-lam*t))/(lam*t))-np.exp(-lam*t))

def phiSim(Nsims, Nt, dt, phi0 = a):
    sim = np.zeros((Nsims, Nt + 1))
    sim[:, 0] = phi0
    for i in range(Nt):
        sim[:, i + 1] = phi((i+1) * dt)
    return sim
        
        
def simOU(x0, alpha, sigma, Nt, dt, Nsims):
    
    X = np.zeros((Nsims, Nt + 1))
    
    X[:, 0] = x0
    brownian = np.random.randn(Nsims, Nt + 1)
    
    for i in range(Nt):
        X[:, i+1] = X[:, i] - alpha * X[:, i] * dt + sigma * brownian[:, i] * np.sqrt(dt)
    return X


def simRt(Nsims, mat, Nt, dt):
    
    x = simOU(x0, alpha, sigma, Nt, dt, Nsims)
    y = simOU(y0, beta, eta, Nt, dt, Nsims)
    phi_t = phiSim(Nsims, Nt, dt, a)
    
    r = phi_t + x + y
    
    return r, x, y
        
        
 
def ESbond(t0, mat, Nsims, Nt):
    
    dt = 1/Nt * mat
    
    x = simOU(x0, alpha, sigma, Nt, dt, Nsims)
    y = simOU(y0, beta, eta, Nt, dt, Nsims)
    phi_t = phiSim(Nsims, Nt, dt, a)
    
    r = phi_t + x + y
    
    Pt0 = np.mean(np.exp(-np.sum(r, axis=1)*dt))
    
    quantiles = np.zeros((3, 3, Nt + 1))
    

    quantiles[0, :, :] = np.quantile(x, [0.1, 0.5, 0.9], axis = 0)
    quantiles[1, :, :] = np.quantile(y, [0.1, 0.5, 0.9], axis = 0)
    quantiles[2, :, :] = np.quantile(r, [0.1, 0.5, 0.9], axis = 0)
        
    return x, y, r, Pt0, quantiles
    
sims = 1000
Nt = 2520
mat = 10

x, y, r, pt0, quantiles = ESbond(0, mat, sims, Nt)

t = np.linspace(0, mat, Nt + 1)

percentage = ["10%", "50%", "90%"]

for j in range(3):
    plt.title("Euler Scheme Estimate of X With Quantiles", fontsize = 16)
    plt.plot(t, quantiles[0, j, :], label = percentage[j])
    plt.legend()
    plt.xlabel("Years")
plt.show()

for j in range(3):
    plt.title("Euler Scheme Estimate of Y With Quantiles", fontsize = 16)
    plt.plot(t, quantiles[1, j, :], label = percentage[j])
    plt.legend()
    plt.xlabel("Years")
plt.show()

for j in range(3):
    plt.title("Euler Scheme Estimate of r With Quantiles", fontsize = 16)
    plt.plot(t, quantiles[2, j, :], label = percentage[j])
    plt.legend()
    plt.xlabel("Years")
plt.show()

#%% question 3

# Monte carlo estimation of yields

def mcYield(Nsims, Nt):
    bonds = []
    y = []
    upper = []
    lower = []
    dt = 1 / 252
    r = ESbond(0, mat, sims, Nt)[2]
    
    
    for i in range(mat * tmonths):
        time = (i+1) * tdays
        bond = np.mean(np.exp(-np.sum(r[:,:time],axis=1)*dt))
        std = np.std(np.log(1/np.exp(-np.sum(r[:,0:time],axis=1)*dt))/((i+1)/12))
        
        bonds.append(bond)
        yt = np.log(1/bond)/((i+1)/tmonths)
        y.append(yt)
        upper.append(yt + std/np.sqrt(Nsims))
        lower.append(yt - std/np.sqrt(Nsims))
        
    return bonds, y, upper, lower

# formula for yield using quad

def integrate(t, mat, sigma, eta, alpha, beta):
    m = sigma * (1 - np.exp(-alpha * (mat - t))) / alpha
    l = eta * (1 - np.exp(-beta * (mat - t))) / beta
    return (m**2 + l**2)

def exactYield(mat):
    bonds = []
    y = []
    
    for i in range(mat * tmonths):
        
        A1 = quad(phi, 0, (i+1) / tmonths, args=())[0]
        A2 = quad(integrate, 0, (i+1)/tmonths, args=((i+1) / tmonths, sigma, eta, alpha, beta))[0]
        
        B = (1 - np.exp(-alpha * ((i+1) / tmonths))) / alpha
        C = (1 - np.exp(-beta * ((i+1) / tmonths))) / beta
        
        bond = np.exp(-A1 + A2/2 - B * x0 - C * y0)
        bonds.append(bond)
        y.append(np.log(1/bond) / ((i+1) / 12))
    
    return bonds, y


estimatedBonds, estimatedYield, upper, lower = mcYield(10_000, Nt)
actualBonds, actualYield = exactYield(10)



t = np.linspace(0, 10, mat * tmonths)
plt.plot(t, estimatedBonds, label = "Monte-Carlo")
plt.plot(t, actualBonds, label = "Actual")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Bond Price")
plt.suptitle("Monte-Carlo Simulation V.S. Analytical Bond Prices", fontsize = 16)
plt.show()


plt.plot(t , estimatedYield, label = "Monte-Carlo")
plt.plot(t, upper, label = "95% CI Upper Bound")
plt.plot(t, lower, label = "95% CI Lower Bound")
plt.plot(t, actualYield, label = "Actual")
plt.legend()
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
plt.suptitle("Monte-Carlo Simulation V.S. Analytical Bond Yields", fontsize = 16)
plt.show()


plt.plot(t , estimatedYield, label = "Monte-Carlo")
plt.plot(t, upper, label = "95% CI Upper Bound")
plt.plot(t, lower, label = "95% CI Lower Bound")
plt.plot(t, actualYield, label = "Actual")
plt.legend()
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
plt.suptitle("Monte-Carlo Simulation V.S. Analytical Bond Yields", fontsize = 16)
plt.ylim((0.0327, 0.0333))
plt.xlim((3,5))
plt.show()

#%% alpha changing

x0 = -0.005
alpha_list = [0.1, 0.5, 1, 2, 5]
sigma = 0.01
y0 = 0.005
beta = 1
eta = 0.005

plt.suptitle(r"Bond Yields With Varying $\alpha$", fontsize = 16)
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
for i in alpha_list:
    alpha = i
    actualBonds, actualYield = exactYield(10)
    plt.plot(t, actualYield, label = r"$\alpha = $" + str(i))
plt.legend()
plt.show()

#%% beta changing

x0 = -0.005
alpha = 3
sigma = 0.01
y0 = 0.005
beta_list = [0.1, 0.5, 1, 2, 5]
eta = 0.005

plt.suptitle(r"Bond Yields With Varying $\beta$", fontsize = 16)
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
for i in beta_list:
    beta = i
    actualBonds, actualYield = exactYield(10)
    plt.plot(t, actualYield, label = r"$\beta = $" + str(i))
plt.legend()
plt.show()


#%% sigma changing

x0 = -0.005
alpha = 3
sigma_list = [0.005, 0.01, 0.1, 0.2, 0.3]
y0 = 0.005
beta = 1
eta = 0.005

plt.suptitle(r"Bond Yields With Varying $\sigma$", fontsize = 16)
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
for i in sigma_list:
    sigma = i
    actualBonds, actualYield = exactYield(10)
    plt.plot(t, actualYield, label = r"$\sigma = $" + str(i))
plt.legend()
plt.show()


#%% eta changing

x0 = -0.005
alpha = 1
sigma = 0.01
y0 = 0.005
beta = 1
eta_list = [0, 0.025, 0.05, 0.1, 0.2]

plt.suptitle(r"Bond Yields With Varying $\eta$", fontsize = 16)
plt.xlabel("Years")
plt.ylabel(r"$y_0(t)$")
for i in eta_list:
    eta = i
    actualBonds, actualYield = exactYield(10)
    plt.plot(t, actualYield, label = r"$\eta = $" + str(i))
plt.legend()
plt.show()

#%% question 5

tenor = np.arange(3, 6.25, 0.25)
F_list = np.arange(0.027,0.038, 0.001)
mat = 3
    
def t_bond(t, mat, x, y):

    A1 = quad(phi, 0, t, args = ())[0]
    A2 = quad(integrate, 0, t, args=(sigma, eta, alpha, beta, mat))[0]
    A = -A1 + A2 / 2
    
    B = (1 - np.exp(-alpha * (mat - t))) / alpha
    
    C = (1 - np.exp(-beta * (mat - t))) / beta

    xt = x[:, t * 252]
    yt = y[:, t * 252]
    
    bonds = np.exp(A - B * xt - C *yt)

    return bonds

def swapRate(x, y, t, tenor):
    
    bonds = np.zeros(tenor.shape)
    
    for i, tau in enumerate(tenor):
        bonds[i] = t_bond(t, mat, x, y).mean()
        
    d_tau = tenor[1:] - tenor[0:-1]
    annuity = np.sum(d_tau * bonds[1:])
    swapRate = (bonds[0] - bonds[-1])/annuity
    return swapRate, annuity

def marketSwap(int_r, t, mat, tenor, F):
    
    bonds = np.zeros((int_r.shape[0], len(tenor)))
    
    for i, tau in enumerate(tenor):
        bonds[:, i] = np.exp(-(int_r[:, int(tau * 252)] - int_r[:, mat * 252]))
    d_tau = tenor[1:] - tenor[0:-1]
    AT = np.sum(d_tau * bonds[:, 1:], axis = 1)
    ST = (bonds[:,0]-bonds[:,-1])/AT

    fixed = np.exp(-(int_r[:,mat * 252] - int_r[:, int(t * 252)]))
    Vt = fixed * AT * np.maximum(ST - F, 0)
    return np.mean(Vt)

def Swaption(x, y, t, mat, tenor, F, int_r):
    
    ST, AT  = swapRate(x, y, mat, tenor)
    fixed = np.mean(np.exp(-(int_r[:, mat * 252] - int_r[:, int(t * 252)])))
    return max(ST - F, 0) * AT * fixed

def implied_vol(St, At, F, mat, marketPrice):
    
    def swaption(omega):
        d1 = (np.log(St/F) + omega / 2) / omega
        d2 = (np.log(St/F) - omega / 2) / omega
        price = At * (St * norm.cdf(d1) - F * norm.cdf(d2))
        fx = price - marketPrice
        return fx

    estimate = root(swaption, 0.034)
    vol = np.min(estimate.x)/np.sqrt(mat)
    
    return vol


x, y, r, Pt0, quantiles = ESbond(0, mat, sims, Nt)

int_r = np.zeros(r.shape)
int_r[:,0]=0
dt = 1/252

for i in range(1,r.shape[1]):
    int_r[:,i] = r[:,i]*dt + int_r[:,i-1] 


S0, A0 = swapRate(x, y, 0, tenor)
S3, A3 = swapRate(x, y, 3, tenor)



V0 = Swaption(x , y, 0, 3, tenor, S0, int_r)

Vt_list = np.full_like(F_list, np.nan)
vol_list = np.full_like(F_list, np.nan)

for i in range(len(F_list)):
    print(i)
    F = F_list[i]
    #Vt_list[i] = Swaption(a,b,lamda,sigma,eta,alpha, beta, x,y, 0, 3, tenor, F, int_r)
    Vt_list[i] = marketSwap(int_r, 0, mat, tenor, F)
    vol_list[i] = implied_vol(S3, A3, F, mat, Vt_list[i])

plt.plot(F_list, Vt_list)
plt.xlabel("Strike")
plt.ylabel("Swaption Price")
plt.title("Swaption Price at Various Strikes", fontsize = 16)
plt.show()


plt.plot(F_list, vol_list)
plt.xlabel("Strike Interest Rate")
plt.ylabel("Black Implied Volitility")
plt.title("Strike Interest Rate vs Black Implied Volitility", fontsize = 16)
plt.show()