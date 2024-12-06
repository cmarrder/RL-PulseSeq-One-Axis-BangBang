import numpy as np

def kappaPlus1(centerTimes, eta1, maxTime):
    return centerTimes + eta1 * np.sin(1 * np.pi * centerTimes / maxTime)

def sinHarmonic1Plus(centerTimes, eta1, maxTime):
    actionInterval = 4 * np.pi
    wbTimes = centerTimes / maxTime * actionInterval
    return (wbTimes + eta1 * np.sin(wbTimes * 1 / 4)) / actionInterval * maxTime

T = 10
eta_kappa = 0.01
eta_harmonic = 0.2
N = 4
pulseTimes = (np.arange(1, N+1) / (N+1)) * T

print("kappaPlus1")
print(kappaPlus1(pulseTimes, eta_kappa, T))
print("sinHarmonic1Plus")
print(sinHarmonic1Plus(pulseTimes, eta_harmonic, T))
