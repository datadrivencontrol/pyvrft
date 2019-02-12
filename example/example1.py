# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:37:20 2019
@author: Emerson Boeira
"""
"""
Testing the function for the inversion of the linear system
"""
#%%Header: import python libraries

import numpy as np #important package for scientific computing: it has important array features
import scipy.signal as sig # signal processing library
import matplotlib.pyplot as plt # library to plot graphics
import vrft # import vrft

#%% First step: defining the system that we want to apply the algorithm
# the SISO system will be defined in a transfer function representation

# numerator and denominator of the discrete-time transfer function
numG=np.array([0.5])
# testing with non-minimun phase
#numG=-np.array([1,-1.2])
denG=np.array([1,-1.3,0.4])
# sampling time
Ts=1

# calculate the state-space model from the numerator and denominator of the transfer function
A,B,C,D=sig.tf2ss(numG, denG)

# calculate system's dimensions: number of states, number of inputs and number of outputs
n=A.shape[0] #number of states
m=B.shape[1] #number of inputs
p=C.shape[0] #number of outputs

#%% Defining the input signal to be applied in the system

# samples of the input signal
N=150
# discrete time vector of the simulation
t=np.linspace(0,N-1,N) #linspace(start,stop,numberofpoints)
# pushing the vector to have the specified dimensions
t.shape=(1,N)
# using a square wave
# defining the period of the square wave
ts=100
fs=1/ts
# finally, defining the square wave using the function square
u=0.5-0.5*sig.square(2*np.pi*fs*t)

# testing another arbitrary input
#u=sig.gausspulse(t,fc=0.03)

# testing another arbitrary input: white noise
#u=np.random.normal(0,1,N)
#u.shape=(1,N)

# plotting the input signal
plt.figure()
plt.step(t.T,u.T)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('u(t)')
plt.show()

#%% Simulating the linear system

# pre-allocating the variables
x=np.zeros((n,N)) # for simplicity, we'll consider x(0)=0
y=np.zeros((p,N))

# calculate the first value for the output (t=0)
y[:,0]=C@x[:,0]+D@u[:,0]
# calculate the states and the output of the system (starting at t=1) with a loop
for k in range(0, N-1):
    x[:,k+1]=A@x[:,k]+B@u[:,k]
    y[:,k+1]=C@x[:,k+1]+D@u[:,k+1]
    
# plotting the output signal
plt.figure()
plt.step(t.T,y.T)
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('y(t)')
plt.show()

#%% Calculate the input signal from the given system and the output signal

uhat,tt=vrft.stblinvlinsys(A,B,C,D,y,t)

# plotting the calculated output
plt.figure()
plt.plot(tt.T,uhat.T,'r')
plt.plot(t.T,u.T,'k-.')
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('uhat(t)')
plt.show()

# comparing the real input vector with the calculated by the inverse algorithm
Nn=np.shape(uhat)[1]
err=np.sum((uhat-u[:,0:Nn])**2)
print(err)