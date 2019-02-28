#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:45:33 2018
@authors: Diego Eckhard and Emerson Boeira
"""
#%% Header: import libraries

from scipy import signal
import numpy as np
import vrft

#%% Functions
   
def filter(G,u):
    
    n=len(G)
    m=len(G[0])
    y=np.zeros((len(u),n))
    
    for i in range(0,n):
        for j in range (0,m):
            if (G[i][j]!=0):
                t,v = signal.dlsim(G[i][j],u[:,j])
                y[:,i]=y[:,i]+v[:,0]
    return y

def filtra_all(G,u):
    
    if (G!=0):    
        y=np.empty(np.shape(u))
        for i, col in enumerate(u.T):
            t,v = signal.dlsim(G,col)
            y[:,i]=v[:,0]
        return y
    else:        
        return np.zeros(np.shape(u))

def design(u,y1,y2,Td,C,L):

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
    r1v,_=vrft.stbinv(Atd,Btd,Ctd,Dtd,y1.T,t)
    r1v=r1v.T
    # calculates the virtual reference for the second data set (instrumental variable)
    r2v,_=vrft.stbinv(Atd,Btd,Ctd,Dtd,y2.T,t)
    r2v=r2v.T
    # remove the last samples of y, to match the dimension of the virtual reference
    # number of samples used in the method
    N=r1v.shape[0]
    y1=y1[0:N,:]
    y2=y2[0:N,:]
    # virtual error
    e1=r1v-y1
    e2=r2v-y2
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
                phi_ij=vrft.filter(C[i][j],e1[:,j:j+1])
                csi_ij=vrft.filter(C[i][j],e2[:,j:j+1])
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
            phivrf[N*j:N*(j+1),pacc:pacc+p_i]=vrft.filtra_all( L[j][i],phi_iN ) # the index are correct, despite the inversion :)
            csivrf[N*j:N*(j+1),pacc:pacc+p_i]=vrft.filtra_all( L[j][i],csi_iN ) # instrumental variable
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

    return p
    
    # Filter Signals 
    # E1 normal experiment
    # E2 instrumental variable
    
#    
#    E1=[]
#    E2=[]
#
#    parametros=0;
#
#    for i in range (0,n):
#        E1.append([])
#        E2.append([])
#        for j in range (0,n):
#            if len(C[i][j])>0:
#                E1[i].append( filter(C[i][j],e1[:,j:j+1]) )
#                E2[i].append( filter(C[i][j],e2[:,j:j+1]) )
#                parametros=parametros+len(C[i][j]);
#            else:
#                E1[i].append( np.empty(shape=(0,0)) )
#                E2[i].append( np.empty(shape=(0,0)) )
#                
#
#    # Filter signals and make ZY matrices
#    Z=np.zeros((parametros,parametros))
#    Y=np.zeros((parametros,1))
#    total=0
#    
#    for i in range (0,n):    
#        EE1=np.zeros((N,parametros))
#        EE2=np.zeros((N,parametros))
#        for j in range (0,n):
#            if E1[i][j].shape[1]>0:
#                par=E1[i][j].shape[1]
#                EE1[:,total:total+par]=E1[i][j] # monta [phi_i(1) phi_i(2)... phi_i(N); 0 0 ... 0 ]^T
#                EE2[:,total:total+par]=E2[i][j]
#            else:
#                par=0
#            total=total+par
#        Z=Z+np.dot(EE1.T,EE2)
#        Y=Y+np.dot(EE1.T,u[:,i:i+1])
#
#    # Compute controller parameters
#    p=np.dot(np.linalg.inv(Z),Y)
#    return p