# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:37:39 2019
@authors: Diego Eckhard and Emerson Boeira
"""
"""
Testing the vrft on a SISO example
"""
#%% Header: importing python libraries

import numpy as np # important package for scientific computing
from scipy import signal # signal processing library
import matplotlib.pyplot as plt # library to plot graphics
import vrft # implementation of vrft

#%% First step: defining the process model, the noise model, the reference model, and the controller class

# declaration of the transfer fuctions that compose the process G(z)
G11=signal.TransferFunction([1],[1,-0.9],dt=1)
G=[[G11]]
# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a warning!!

# declaration of the transfer fuctions that compose the noise model H(z)
H11=signal.TransferFunction([1],[1],dt=1)
# organizing the system H(z) in a python list
H=[[H11]]

# declaration of the transfer fuctions that compose the reference model Td(z)
Td11=signal.TransferFunction([0.2],[1,-0.8],dt=1)
# organizing the MIMO reference model Td(z) in a python list
Td=[[Td11]]

# choosing the VRFT method filter
L11=signal.TransferFunction([0.25],[1,-0.75],dt=1)
# organizing the filter as a list
L=[[L11]]
#L=Td

# defining the controller structure that will be used in the method
Cp=[[signal.TransferFunction([1],[1],dt=1)]] # proportional controller structure
Cpi=[[signal.TransferFunction([1, 0],[1, -1],dt=1)] , [signal.TransferFunction([1],[1, -1],dt=1)]] # PI controller structure
# assembling the controller structure
C=[[Cpi]] # in this example, we choosed a decentralized PI controller

#%% Simulating the open loop system to obtain the data for the VRFT

# samples of the input signal
N=350
#N=10
# discrete time vector of the simulation
t=np.linspace(0,N-1,N) # linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape=(1,N)

# using a square wave for both inputs
# defining the period of the square wave
ts=N
fs=1/ts
# finally, defining the square wave using the function signal.square()
u=0.5-0.5*signal.square(2*np.pi*fs*t).T
## testing another arbitrary input: gaussian pulse
#u=signal.gausspulse(t,fc=0.003).T
#IMPORTANT: in our toolbox, we decided to organize the input and output signals as an matrix (N,n), where N=number of data samples, n=number of inputs and outputs

# plotting the input signal
plt.figure()
plt.step(t.T,u)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('u(t)')
plt.show()

# calculating the output of the system
yu=vrft.filter(G,u)
# add noise to the output
# variance of the whie noise signal
sigma2_e1=0
# creating noise vectors
w=np.random.normal(0,np.sqrt(sigma2_e1),N)
# pushing the dimensions to match our signals
w.shape=(N,1)
# filtering white noise by H(z)
v=vrft.filter(H,w)
# real (measured) output
y=yu+v;

# plotting the output signal
plt.figure()
plt.step(t.T,y)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('y(t)')
plt.show()

# design the controller using the VRFT method
p=vrft.design(u,y,y,Td,C,L)
print("p=",p)