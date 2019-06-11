# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:35:53 2019
@authors: Diego Eckhard and Emerson Boeira
"""
"""
Testing the vrft on a SISO example with a non-minimum phase reference model - The inversion algorithm should fail!
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% First step: defining the process model, the noise model, the reference model, and the controller class

# declaration of the transfer fuction of the process G(z)
G = signal.TransferFunction([-1, +1.2], [1, -1.7, 0.72], dt=1)
# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a warning!!

# declaration of the transfer fuction of the noise model H(z)
H = signal.TransferFunction([1], [1], dt=1)

# declaration of the transfer fuction of the reference model Td(z)
Td = signal.TransferFunction([-0.4, 0.48], [1, -1.4, 0.48], dt=1)

# choosing the VRFT method filter
# L=signal.TransferFunction([0.25],[1,-0.75],dt=1)
# a simple choice
L = Td

# defining the controller structure that will be used in the method
Cp = [[signal.TransferFunction([1], [1], dt=1)]]  # proportional controller structure
Cpi = [
    [signal.TransferFunction([1, 0], [1, -1], dt=1)],
    [signal.TransferFunction([1], [1, -1], dt=1)],
]  # PI controller structure
Cpid = [
    [signal.TransferFunction([1, 0, 0], [1, -1, 0], dt=1)],
    [signal.TransferFunction([1, 0], [1, -1, 0], dt=1)],
    [signal.TransferFunction([1], [1, -1, 0], dt=1)],
]  # PID controller structure
# assembling the controller structure
C = Cpid  # in this example, we choosed a PI controller

#%% Simulating the open loop system to obtain the data for the VRFT

# samples of the input signal
N = 150
# discrete time vector of the simulation
t = np.linspace(0, N - 1, N)  # linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape = (1, N)

# using a square wave for both inputs
# defining the period of the square wave
ts = N
fs = 1 / ts
# finally, defining the square wave using the function signal.square()
u = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t).T
## testing another arbitrary input: gaussian pulse
# u=signal.gausspulse(t,fc=0.003).T
# IMPORTANT: in our package, we decided to organize the input and output signals as an matrix (N,n), where N=number of data samples, n=number of inputs and outputs

# plotting the input signal
plt.figure()
plt.step(t.T, u)
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("u(t)")
plt.show()

# calculating the output of the system
yu = vrft.filter(G, u)
# add noise to the output
# variance of the whie noise signal
sigma2_e1 = 0
# creating noise vectors
w = np.random.normal(0, np.sqrt(sigma2_e1), N)
# pushing the dimensions to match our signals
w.shape = (N, 1)
# filtering white noise by H(z)
v = vrft.filter(H, w)
# real (measured) output
y = yu + v

# plotting the output signal
plt.figure()
plt.step(t.T, y)
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("y(t)")
plt.show()

# design the controller using the VRFT method
p = vrft.design(u, y, y, Td, C, L)
print("p=", p)
