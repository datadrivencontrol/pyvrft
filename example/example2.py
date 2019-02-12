# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:36:13 2019
@author: Emerson Boeira
"""
"""
Testing the inversion algorithm on a MIMO system
"""
#%%Header: import python libraries

import numpy as np # important package for scientific computing: important array features
import scipy.signal as sig # signal processing library
import matplotlib.pyplot as plt # library to plot graphics
import vrft # import vrft

#%% First step: defining the MIMO system that we want to apply the algorithm

# for now, the system will be defined in the state-space representation
# in this file, there is some examples of systems to test the algorithm

# SYSTEM 1
# dynamic system matrix
A=np.array([[0,-0.855,0],[1,1.85,0],[0,0,0.95]])
# input matrix
B=np.array([[0.04,1],[0,-1],[0.28,0.08]])
# output matrix
C=np.array([[0,1,0],[0,0,0.25]])
# feedforward matrix
D=np.zeros((2,2))

## SYSTEM 2 - WITH A NON-MINIMUM PHASE TRANSMISSION ZERO
## dynamic system matrix
#A=np.array([[1.7,-0.72,0,0],[1,0,0,0],[0,0,0.9,0],[0,0,0,0.9]])
## input matrix
#B=np.array([[1,0],[0,0],[1,0],[0,1]])
## output matrix
#C=np.array([[1,0,0,0.6],[0,0,1,0.2]])
## feedforward matrix
#D=np.zeros((2,2))

## SYSTEM 3 - WITH MORE INPUTS THAN OUTPUTS - THE ALGORITHM FAILS!
## dynamic system matrix
#A=np.array([[0.9,0,0,0,0,0],[0,0.8,0,0,0,0],[0,0,0.8,0,0,0],[0,0,0,1.7,-0.72,0],[0,0,0,1,0,0],[0,0,0,0,0,0.9]])
## input matrix
#B=np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,1]])
## output matrix
#C=np.array([[1,0,0,0,1,0],[0,-1,1,0,0,1]])
## feedforward matrix
#D=np.zeros((2,3))

## SYSTEM 4 - WITH MORE OUTPUTS THAN INPUTS
## dynamic system matrix
#A=np.array([[0.9,0],[0,0.75]])
## input matrix
#B=np.array([[2],[2]])
## output matrix
#C=np.array([[1,0],[0,3.5]])
## feedforward matrix
#D=np.zeros((2,1))

# calculate systems dimensions: number of states, number of inputs and number of outputs
n=A.shape[0] #number of states
m=B.shape[1] #number of inputs
p=C.shape[0] #number of outputs

#%% Defining the input signal to be applied in the system

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
#u1=0.5-0.5*sig.square(2*np.pi*fs*t)
#u2=0.5-0.5*sig.square(2*np.pi*fs*t-3*np.pi/2)

# testing another arbitrary inputs: gaussian pulse
u1=sig.gausspulse(t,fc=0.003)
u2=sig.gausspulse(t,fc=0.002)
#u3=sig.gausspulse(t,fc=0.0035)

# testing another arbitrary inputs: white noise
#u1=np.random.normal(0,1,N)
#u2=np.random.normal(0,1,N)
#u1.shape=(1,N)
#u2.shape=(1,N)

u=np.concatenate((u1,u2),axis=0)
#u=np.concatenate((u1,u2,u3),axis=0)

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
plt.plot(tt.T,uhat.T)
plt.plot(t.T,u.T,'k-.')
plt.grid(True)
plt.xlabel('time (t)')
plt.ylabel('uhat(t)')
plt.show()