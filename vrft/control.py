#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:45:33 2018
@authors: Diego Eckhard and Emerson Boeira
"""
#%% Header: import libraries

from scipy import signal # signal processing library
import numpy as np # important package for scientific computing
import vrft # vrft package

#%% Functions
   
def filter(G,u):
    # function used to filter the signals in a MIMO structure, as we defined in our toolbox
    # number of outputs
    n=len(G)
    # number of inputs
    m=len(G[0])
    # preallocating the output
    y=np.zeros((len(u),n))
    # loop to calculate each output signal
    for i in range(0,n):
        for j in range (0,m):
            if (G[i][j]!=0):
                t,v = signal.dlsim(G[i][j],u[:,j])
                y[:,i]=y[:,i]+v[:,0]
    # return the output (filtered) signal
    return y

def colfilter(G,u):
    # function that filter every column of u with the same filter
    
    # test if the transfer function is not zero
    # preallocating the output
    y=np.zeros((np.shape(u)))
    if (G!=0):    
        # loop for each column of u
        for i, col in enumerate(u.T):
            t,v = signal.dlsim(G,col)
            y[:,i]=v[:,0]
    return y

def design(u,y,y_iv,Td,C,L):
    # function that implements the Unbiased MIMO VRFT method

    # number of data samples
    N=len(u)
    # number of inputs/outputs of the system
    n=len(Td)
    # creates a dummy time vector, necessary for the function that calculates the virtual reference
    t=np.linspace(0,N-1,N) #linspace(start,stop,numberofpoints)
    # pushing the vector to have the specified dimensions
    t.shape=(1,N)
    
    # Filter u
    uf=vrft.filter(L,u)
    
    # transformation of Td from the MIMO transfer function list structure to a state-space model
    Atd,Btd,Ctd,Dtd=vrft.mtf2ss(Td)
    # calculates the virtual reference for the first data set
    rv,_,flagvr=vrft.stbinv(Atd,Btd,Ctd,Dtd,y.T,t)
    rv=rv.T
    # calculates the virtual reference for the second data set (instrumental variable)
    rv_iv,_,_=vrft.stbinv(Atd,Btd,Ctd,Dtd,y_iv.T,t)
    rv_iv=rv_iv.T
    
    # test if the inversion algorithm was succesful
    if flagvr==0:
        # if flagvr=0, then, the inversion algorithm was succesful
    
        # remove the last samples of y, to match the dimension of the virtual reference
        # number of samples used in the method
        N=rv.shape[0]
        y=y[0:N,:]
        y_iv=y_iv[0:N,:]
        # virtual error
        ebar=rv-y
        ebar_iv=rv-y_iv
        # remove the last samples of the input (to match the dimension of the virtual error)
        uf=uf[0:N,:]
        
        # calculating the number of parameters of each controller and organizing it as a matrix
        # preallocating
        nbpar=np.zeros((n,n))
        # loop
        for i in range (0,n):
            for j in range (0,n):
                nbpar[i][j]=len(C[i][j])
            
        # total number of parameters (casting to integer)
        p_tot=int(np.sum(nbpar))
    
        # assembling the matrix phi_N=[phi_1N^T phi_2N^T...phi_nN^T]       
        # preallocating
        phi_N=np.zeros((N,p_tot))
        csi_N=np.zeros((N,p_tot))
        pacc=0
        # loop that organizes the matrix phi_N=[phi_1N^T phi_2N^T...phi_nN^T]       
        for i in range (0,n):
            for j in range (0,n):
                if len(C[i][j])>0:
                    # calculating phi_ij(t)^T
                    phi_ij=vrft.filter(C[i][j],ebar[:,j:j+1])
                    # instrumental variable
                    csi_ij=vrft.filter(C[i][j],ebar_iv[:,j:j+1])
                    # number of parameters in Cij(z)
                    pij=int(nbpar[i][j])
                    # assembling phi_N
                    phi_N[:,pacc:pacc+pij]=phi_ij
                    csi_N[:,pacc:pacc+pij]=csi_ij
                    # parameter accumulator
                    pacc=pacc+pij
                
        # using the accumulator again
        pacc=0                
        # loop that organize the phi_vrf matrix (with the filter)
        # preallocating
        phivrf=np.zeros((N*n,p_tot))
        csivrf=np.zeros((N*n,p_tot))
        # loop
        for i in range (0,n):
            for j in range (0,n):
                # number of parameters regarding the signal ui:
                p_i=int(np.sum(nbpar[i])) # casting to integer as well
                # separates the phi_iN signal
                phi_iN=phi_N[:,pacc:pacc+p_i]
                csi_iN=csi_N[:,pacc:pacc+p_i]
                # using the MIMO filter L(q)
                phivrf[N*j:N*(j+1),pacc:pacc+p_i]=vrft.colfilter( L[j][i],phi_iN ) # the index are correct, despite the inversion :)
                csivrf[N*j:N*(j+1),pacc:pacc+p_i]=vrft.colfilter( L[j][i],csi_iN ) # instrumental variable
            # acumulating the parameters
            pacc=pacc+p_i
        
        # reorganizing the uf vector (stacking)
        # preallocating
        Uf=np.zeros((N*n,1))
        # loop
        for i in range (0,n):
            Uf[N*i:N*(i+1)]=uf[:,i:i+1]
        
        # compute parameters
        Z=csivrf.T@phivrf
        Y=csivrf.T@Uf
        p=np.linalg.inv(Z)@Y
        
        # returning the parameter vector
        return p
    
    elif flagvr==1:
        # if flagvr=1, then, it was not possible to calculate the inverse of the reference model
        print("It was not possible to calculate the virtual reference.")
        # return an empty parameter vector
        p=np.empty((0,0))
        return p
        
    elif flagvr==2:
        # if flagvr=2, the inverse of the reference model is unstable. VRFT method aborted!
        print("The inverse of the reference model Td(z) is unstable. It is not recommended to proceed with the VRFT method, so the algorithm was aborted!")
        # return an empty parameter vector
        p=np.empty((0,0))
        return p