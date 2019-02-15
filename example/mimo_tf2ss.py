# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 19:45:59 2019
@authors: Diego Eckhard and Emerson Boeira
"""
"""
Function used to implement the transformation from a MIMO transfer function model to a state space model
"""
#%% Header: importing python libraries

import numpy as np # important package for scientific computing
from scipy import signal # signal processing library
import matplotlib.pyplot as plt # library to plot graphics
import vrft # import vrft

#%% Defining the system in the transfer function list structure (as we have been working)

# declaration of the transfer fuctions that compose the MIMO process G(z)
G11 = signal.TransferFunction([1],[1,-0.9],dt=1)
G12 = signal.TransferFunction([0.6],[1,-0.7],dt=1)
G21 = 0
G22 = signal.TransferFunction([1.4],[1,-0.65],dt=1)
# organizing the MIMO system G(z) in a python list
G=[[G11,G12],[G21,G22]]
# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a WARNING!!

# %% TESTING THE IMPLEMENTATION OF THE FUNCTION

# using the function
Ass,Bss,Css,Dss=vrft.mtf2ss(G)
            
#%% Test if the both systems are equivalent

# DEFINIG AN INPUT SIGNAL TO THE TEST
# samples of the input signal
N=500
# discrete time vector of the simulation
t=np.linspace(0,N-1,N) # linspace(start,stop,numberofpoints)
# pushing the time vector to have the specified dimensions
t.shape=(1,N)

# using a square wave for both inputs
# defining the period of the square wave
ts=N
fs=1/ts
# finally, defining the square wave using the function signal.square()
#u1=0.5-0.5*signal.square(2*np.pi*fs*t).T
#u2=0.5-0.5*signal.square(2*np.pi*fs*t-3*np.pi/2).T
## testing another arbitrary inputs: gaussian pulse
u1=signal.gausspulse(t,fc=0.003).T
u2=signal.gausspulse(t,fc=0.002).T
# concatenating the signals
u=np.concatenate((u1,u2),axis=1)

# calculating the output of the MIMO system by the filtering option: TRANSFER FUNCTION MODEL
yg=vrft.filter(G,u)
# calculating the output of the MIMO system by the simulation of the STATE-SPACE MODEL
# calculating the number of outputs
p=len(G)
# calculating the number of inputs
m=len(G[0])
# number of states
nss=Ass.shape[0]
# preallocating the variables
x=np.zeros((nss,N)) # for simplicity, we'll consider x(0)=0
yss=np.zeros((p,N))
# calculate the first value for the output (t=0)
yss[:,0]=Css@x[:,0]+Dss@u[0,:].T
## calculate the states and the output of the system (starting at t=1) with a loop
for k in range(0, N-1):
    x[:,k+1]=Ass@x[:,k]+Bss@u[k,:].T
    yss[:,k+1]=Css@x[:,k+1]+Dss@u[k+1,:].T

# plotting the output signal
plt.figure()
plt.step(t.T,yg)
plt.plot(t.T,yss.T,'k-.')
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('y(t)')
plt.show()