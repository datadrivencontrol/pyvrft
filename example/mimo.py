# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:24:38 2019
@authors: Diego Eckhard and Emerson Boeira

Testing the vrft function on an example
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # implementation of vrft

#%% Simulating the open loop system to obtain the data for the VRFT

# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a warning!!
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
N = 350
# discrete time vector of the simulation
t = np.linspace(0, N - 1, N)  # linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape = (1, N)

# using a square wave for both inputs
# defining the period of the square wave
ts = N
fs = 1 / ts
# finally, defining the square wave using the function signal.square()
u1 = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t).T
u2 = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t - 3 * np.pi / 2).T

# concatenating the signals
u = np.concatenate((u1, u2), axis=1)
# IMPORTANT: in our package, we decided to organize the input and output signals as an matrix (N,n), where N=number of data samples, n=number of inputs and outputs

# calculating the output of the MIMO system
yu = vrft.filter(G, u)
# add noise to the output
# variance of the whie noise signals (in this example we choose sigma2=0, i.e. we dont have any noise)
sigma2_e1 = 0.001
sigma2_e2 = 0.001
# creating noise vectors
w1 = np.random.normal(0, np.sqrt(sigma2_e1), N)
w2 = np.random.normal(0, np.sqrt(sigma2_e2), N)
# pushing the dimensions to match our signals
w1.shape = (N, 1)
w2.shape = (N, 1)
# concatenating noise signals
w = np.concatenate((w1, w2), axis=1)
# real (measured) output
y = yu + w

# plotting the signals
plt.figure()
plt.step(t.T, u)
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("u(t)")
plt.show()

# plotting the output signal
plt.figure()
plt.step(t.T, y)
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("y(t)")
plt.show()

#%% CONTROL

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
L11 = signal.TransferFunction([0.25], [1, -0.75], dt=1)
L12 = signal.TransferFunction([0.4], [1, -0.65], dt=1)
L21 = signal.TransferFunction([0.1], [1, -0.75], dt=1)
L22 = signal.TransferFunction([0.2], [1, -0.6], dt=1)
# organizing the MIMO filter as a list
L = [
     [L11, L12], 
     [L21, L22]
]

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

# design the controller using the VRFT method
p = vrft.design(u, y, y, Td, C, L)
print("p=", p)