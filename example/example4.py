# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:53:47 2019
@author: Emerson
"""
"""
Testing the inversion algorithm on a MIMO system and the transformation of a MIMO transfer function to the state-space representation
"""
#%%Header: import python libraries

import numpy as np # important package for scientific computing: important array features
import scipy.signal as signal # signal processing library
import matplotlib.pyplot as plt # library to plot graphics
import vrft # import vrft

#%% First step: defining the MIMO system

# in this example, we'll define the MIMO system in structure of transfer function lists

# SYSTEM 1
# declaration of the transfer fuctions that compose the MIMO process G(z)
G11 = signal.TransferFunction([0.04],[1,-1.85,0.855],dt=1)
G12 = signal.TransferFunction([-1,1],[1,-1.85,0.855],dt=1)
G21 = signal.TransferFunction([0.2],[1,-0.95],dt=1)
G22 = signal.TransferFunction([0.15],[1,-0.95],dt=1)

### SYSTEM 2 - WITH A NON-MINIMUM PHASE TRANSMISSION ZERO
## transfer functions that compose the process
#G11=signal.TransferFunction([1,0],[1,-1.7,0.72],dt=1)
#G12=signal.TransferFunction([0.6],[1,-0.9],dt=1)
#G21=signal.TransferFunction([1],[1,-0.9],dt=1)
#G22=signal.TransferFunction([0.2],[1,-0.9],dt=1)

## SYSTEM 3 - WITH MORE INPUTS THAN OUTPUTS - THE ALGORITHM FAILS!
#G11=signal.TransferFunction([1],[1,-0.9],dt=1)
#G12=0
#G13=signal.TransferFunction([1],[1,-1.7,0.72],dt=1)
#G21=signal.TransferFunction([-1],[1,-0.8],dt=1)
#G22=signal.TransferFunction([1],[1,-0.7],dt=1)
#G23=signal.TransferFunction([1],[1,-0.9],dt=1)

## SYSTEM 4 - WITH MORE OUTPUTS THAN INPUTS
#G11=signal.TransferFunction([2],[1,-0.9],dt=1)
#G21=signal.TransferFunction([7],[1,-0.75],dt=1)

# organizing the MIMO system G(z) in a python list
G=[[G11,G12],[G21,G22]]
#G=[[G11,G12,G13],[G21,G22,G23]]
#G=[[G11],[G21]]

# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a WARNING!!

#%% Simulating the open loop system

# samples of the input signal
N=1000
# discrete time vector of the simulation
t=np.linspace(0,N-1,N) #linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape=(1,N)

# using a square wave for both inputs
# defining the period of the square wave
ts=N
fs=1/ts
# finally, defining the square wave using the function square
#u1=0.5-0.5*signal.square(2*np.pi*fs*t).T
#u2=0.5-0.5*signal.square(2*np.pi*fs*t-3*np.pi/2).T
### testing another arbitrary inputs: gaussian pulse
u1=signal.gausspulse(t,fc=0.003).T
u2=signal.gausspulse(t,fc=0.002).T
#u3=signal.gausspulse(t,fc=0.0035).T
## testing another arbitrary inputs: white noise
#u1=np.random.normal(0,1,N)
#u2=np.random.normal(0,1,N)
#u1.shape=(N,1)
#u2.shape=(N,1)

# concatenating the signals
u=np.concatenate((u1,u2),axis=1)
#u=np.concatenate((u1,u2,u3),axis=1)
#u=u1
#IMPORTANT: in our toolbox, we decided to organize the input and output signals as an array (N,n), where N=number of data samples, n=number of inputs and outputs

# plotting the input signal
plt.figure()
plt.step(t.T,u)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('u(t)')
plt.show()

# calculating the output of the MIMO system
y=vrft.filter(G,u)

# plotting the output signal
plt.figure()
plt.step(t.T,y)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('y(t)')
plt.show()

#%% Transforming the system to the state-space model

# using the function that transform the MIMO system to a state-space representation
Ass,Bss,Css,Dss=vrft.dmimo_tf2ss(G)

#%% Calculate the input signal from the given system and the output signal

uhat,tt=vrft.stblinvlinsys(Ass,Bss,Css,Dss,y.T,t)

# plotting the calculated output
plt.figure()
plt.plot(tt.T,uhat.T)
plt.plot(t.T,u,'k-.')
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('uhat(t)')
plt.show()