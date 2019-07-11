# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 14:37:39 2019
@authors: Diego Eckhard and Emerson Boeira
"""
"""
Testing the vrft on a SISO example
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% Simulating the open loop system to obtain the data for the VRFT

# declaration of the SISO transfer fuction of the process G(z)
G = signal.TransferFunction([1], [1, -0.9], dt=1)
# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]
# num=[0,1] produces a warning!

# number of samples
N = 100

# step signal
u = np.ones((N,1))
u[0] = 0
# IMPORTANT: in our package, we decided to organize the input and output signals as a matrix (N,n)
# N=number of data samples, n=number of inputs and outputs

# calculating the output of the system
yu = vrft.filter(G, u)

# add noise to the output
# variance of the whie noise signal
sigma2_e1 = 0.1
# creating noise vector
w = np.random.normal(0, np.sqrt(sigma2_e1), N)
# pushing the dimensions to match our signals
w.shape = (N, 1)

# real (measured) output
y = yu + w

# plot input signal
plt.figure()
plt.plot(u,drawstyle='steps')
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("u(t)")
plt.show()

# plot output signal
plt.figure()
plt.plot(y,drawstyle='steps')
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("y(t)")
plt.show()

#%% CONTROL - VRFT parameters: reference model Td(z), filter L(z), and controller structure

# declaration of the transfer fuction of the reference model Td(z)
Td = signal.TransferFunction([0.2], [1, -0.8], dt=1)

# choosing the VRFT method filter
L = signal.TransferFunction([0.25], [1, -0.75], dt=1)

# defining the controller structure that will be used in the method
C = [
    [signal.TransferFunction([1, 0], [1, -1], dt=1)],
    [signal.TransferFunction([1], [1, -1], dt=1)],
]  # PI controller structure

#%% Design the controller using the VRFT method

# VRFT with least squares
p = vrft.design(u, y, y, Td, C, L)
print("p=", p)