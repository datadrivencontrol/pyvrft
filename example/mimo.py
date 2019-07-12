# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:24:38 2019
@authors: Diego Eckhard and Emerson Boeira

Testing the vrft on a MIMO example with instrumental variable
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% Simulating the open loop system to obtain the data for the VRFT

# IMPORTANT: if the numerator of the transfer function is 1, define it as num=[1], instead of num=[0,1]
# num=[0,1] produces a warning!
# declaration of the transfer fuctions that compose the MIMO process G(z)
G11 = signal.TransferFunction([0.09516], [1, -0.9048], dt=1)
G12 = signal.TransferFunction([0.03807], [1, -0.9048], dt=1)
G21 = signal.TransferFunction([-0.02974], [1, -0.9048], dt=1)
G22 = signal.TransferFunction([0.04758], [1, -0.9048], dt=1)
# organizing the MIMO system G(z) in a python list
G = [
     [G11, G12], 
     [G21, G22]
]

# samples of the input signal
N = 1000
# discrete time vector
t = np.linspace(0, N - 1, N)  # linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape = (1, N)

# using a square wave for both inputs
# defining the period of the square wave
ts = N / 2
fs = 1 / ts
# finally, defining the square wave using the function signal.square()
u1 = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t).T
u2 = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t - 3 * np.pi / 2).T

# concatenating signals
u = np.concatenate((u1, u2), axis=1)
# IMPORTANT: in our package, we decided to organize the input and output signals as a matrix (N,n)
# N=number of data samples, n=number of inputs and outputs

# calculating the output of the MIMO system
yu = vrft.filter(G, u)
# add noise to the output
# variance of the whie noise
sigma2_e1 = 0.001
sigma2_e2 = 0.001
# creating noise vectors for each experiment
# noise of the first experiment
w1a = np.random.normal(0, np.sqrt(sigma2_e1), N)
w2a = np.random.normal(0, np.sqrt(sigma2_e2), N)
# noise of the second experiment
w1b = np.random.normal(0, np.sqrt(sigma2_e1), N)
w2b = np.random.normal(0, np.sqrt(sigma2_e2), N)
# pushing the dimensions to match our signals
w1a.shape = (N, 1)
w2a.shape = (N, 1)
w1b.shape = (N, 1)
w2b.shape = (N, 1)
# concatenating noise signals
wa = np.concatenate((w1a, w2a), axis=1)
wb = np.concatenate((w1b, w2b), axis=1)
# real (measured) output for each experiment
ya = yu + wa
yb = yu + wb

#%% Graphics

# linwidth
lw = 1.5

# plot input signals
plt.figure()
plt.plot(u[:, 0], "b", drawstyle="steps", linewidth=lw, label="u1(t)")
plt.plot(u[:, 1], "r", drawstyle="steps", linewidth=lw, label="u2(t)")
plt.grid(True)
plt.xlabel("time (samples)")
plt.ylabel("u(t)")
plt.xlim(left=0, right=N)
plt.legend(loc="upper left")
# plt.savefig('u_sim.eps', format = 'eps', dpi=600)
plt.show()

# plot output signal
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ya[:, 0], "b", drawstyle="steps", linewidth=lw, label="y1a(t)")
plt.plot(ya[:, 1], "r", drawstyle="steps", linewidth=lw, label="y2a(t)")
plt.grid(True)
plt.ylabel("ya(t)")
plt.xlim(left=0, right=N)
plt.legend(loc="upper left")
plt.subplot(2, 1, 2)
plt.plot(yb[:, 0], "b", drawstyle="steps", linewidth=lw, label="y1b(t)")
plt.plot(yb[:, 1], "r", drawstyle="steps", linewidth=lw, label="y2b(t)")
plt.grid(True)
plt.xlabel("time (samples)")
plt.ylabel("yb(t)")
plt.xlim(left=0, right=N)
plt.legend(loc="upper left")
# plt.savefig('y_sim.eps',format='eps',dpi=600)
plt.show()

#%% Control - VRFT parameters: reference model Td(z), filter L(z), and controller structure

# declaration of the transfer fuctions that compose the MIMO reference model Td(z)
Td11 = signal.TransferFunction([0.25], [1, -0.75], dt=1)
Td12 = 0
Td21 = 0
Td22 = signal.TransferFunction([0.4], [1, -0.6], dt=1)
# organizing the MIMO reference model Td(z) in a python list
Td = [
      [Td11, Td12], 
      [Td21, Td22]
]

# choosing the VRFT method filter
L = Td

# defining the controller structure that will be used in the method
Cpi = [
    [signal.TransferFunction([1, 0], [1, -1], dt=1)],
    [signal.TransferFunction([1], [1, -1], dt=1)],
]  # PI controller structure
# assembling the MIMO controller structure
C = [
     [Cpi, Cpi], 
     [Cpi, Cpi]
]  # in this example, we choosed a full PI controller

#%% Design the controller using the VRFT method

# VRFT with instrumental variables - using output data of both experiments
p = vrft.design(u, ya, yb, Td, C, L)
print("p=", p)