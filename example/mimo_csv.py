#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 3 16:14:00 2019
@authors: Emerson Boeira and Diego Eckhard

Testing the vrft with data from a .csv file
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% Reading the data from the .csv file

# some parameters to read from the .csv using our package
# offset (or number of lines of the header of the .csv file)
offset = 5
# number of inputs and outputs of the system to be controlled
n = 2
# read the data from the csv file using a function available on our package
y, u = vrft.datafromcsv("data.csv", ",", offset, n)

# preprocessing the data
# removing the first Nr samples
Nr = 1000
ya = y[Nr - 1 : -1, :]
ua = u[Nr - 1 : -1, :]

# removing the mean value of the signals
ya = ya - np.mean(y, 0)
ua = ua - np.mean(u, 0)

#%% Graphics

lw = 1.5  # linewidth

# plot input signals
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(u[500:-1, 0], "b", drawstyle="steps", linewidth=lw)
plt.grid(True)
plt.ylabel("u1(t)")
plt.xlim(left=0, right=7500)
plt.subplot(2, 1, 2)
plt.plot(u[500:-1, 1], "b", drawstyle="steps", linewidth=lw)
plt.grid(True)
plt.xlabel("time (s)")
plt.ylabel("u2(t)")
plt.xlim(left=0, right=7500)
# plt.savefig('u_exp.eps',format='eps',dpi=600)
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(y[500:-1, 0], "b", drawstyle="steps", linewidth=lw)
plt.grid(True)
plt.ylabel("y1(t)")
plt.xlim(left=0, right=7500)
plt.subplot(2, 1, 2)
plt.plot(y[500:-1, 1], "b", drawstyle="steps", linewidth=lw)
plt.grid(True)
plt.xlabel("time (s)")
plt.ylabel("y2(t)")
plt.xlim(left=0, right=7500)
# plt.savefig('y_exp.eps',format='eps',dpi=600)
plt.show()

#%% Control - VRFT parameters: reference model Td(z), filter L(z), and controller structure

# declaration of the transfer fuctions that compose the MIMO reference model Td(z)
Td11 = signal.TransferFunction([0.03], [1, -0.97], dt=1)
Td12 = 0
Td21 = 0
Td22 = signal.TransferFunction([0.02], [1, -0.98], dt=1)
# organizing the MIMO reference model Td(z) in a python list
Td = [
      [Td11, Td12], 
      [Td21, Td22]
]

# choosing the VRFT method filter L=Td*(Td-I)
L11 = signal.TransferFunction([-0.03, 0.03], [1, -1.94, 0.9409], dt=1)
L12 = 0
L21 = 0
L22 = signal.TransferFunction([-0.02, 0.02], [1, -1.96, 0.9604], dt=1)
# organizing the MIMO filter L(z) in a python list
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
]  # in this example, a full PI controller is tuned

#%% Design the controller using the VRFT method

# VRFT with least squares - just one set of data
p = vrft.design(ua, ya, ya, Td, C, L)
print("p=", p)
