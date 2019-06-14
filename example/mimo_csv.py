#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:14:00 2019
@authors: Emerson Boeira and Diego Eckhard

Testing the vrft with data read from a .csv file
"""
#%% Header: importing python libraries

from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% Reading the data from the .csv file

# some parameters to read from the .csv using our package
# offset (or number of lines of the Header of the .csv file)
offset = 5
# number of inputs and outputs of the system to be controlled
n = 2
# read the data from the csv file using a function available on our package
ya, ua = vrft.datafromcsv("data_a.csv", ",", offset, n)
yb, ub = vrft.datafromcsv("data_b.csv", ",", offset, n)
# choosing the input
u = ua

# plotting the input signal
plt.figure()
plt.plot(u, drawstyle='steps')
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("u(t)")
plt.show()

# plotting y1(t) for both experiments
plt.figure()
plt.plot(ya[:, 0], drawstyle='steps')
plt.plot(yb[:, 0], drawstyle='steps')
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("y1(t)")
plt.show()

# plotting y2(t) for both experiments
plt.figure()
plt.plot(ya[:, 1], drawstyle='steps')
plt.plot(yb[:, 1], drawstyle='steps')
plt.grid(True)
plt.xlabel("time (t)")
plt.ylabel("y2(t)")
plt.show()

#%% VRFT parameters: reference model: Td(z), filter: L(z), and controller structure

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
    [Cpi, Cpi],
]  # in this example, we choosed a decentralized PI controller

#%% Design the controller using the VRFT method

p = vrft.design(u, ya, yb, Td, C, L)
print("p=", p)