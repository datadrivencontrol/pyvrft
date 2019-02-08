#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:45:33 2018
@authors: Diego Eckhard and Emerson Boeira
"""

from scipy import signal
import numpy as np


def inverte(G):
    return 0
    
    
AA=np.array([[0.9, 0],[1, 0.9]])
BB=np.array([[1],[0]])
CC=np.array([[1, 0]])
DD=np.array([[1]])

GG=signal.StateSpace(AA,BB,CC,DD,dt=1)
  

u1=np.zeros((10,1))
u1[1]=1

tt,yy,xx=signal.dlsim(GG,u1)
  
    
def filtra(G,u):
    
    n=len(G)
    m=len(G[0])
    y=np.zeros((len(u),n))
    
    for i in range(0,n):
        for j in range (0,m):
            if (G[i][j]!=0):
                t,v = signal.dlsim(G[i][j],u[:,j]);
                y[:,i]=y[:,i]+v[:,0]
    return y

def filtra_all(G,u):
    
    if (G!=0):
    
        y=np.empty(np.shape(u))
        
        for i, col in enumerate(u.T):
            t,v = signal.dlsim(G,col);
            y[:,i]=v[:,0]
        return y

    else:
        
        return np.zeros(np.shape(u))

def vrft_mimo(u,y1,y2,Td,C,L):

    N=len(u);
    n=len(Td);
    
    # Filter u
#    u=filtra(L,u);
    u=filtra(L,u);
    
    # Compute virtual error and filter S=(eye(n)-Td);
    e1=y1-filtra(Td,y1);
    e2=y2-filtra(Td,y2);
 
    
    # Filter Signals 
    # E1 normal experiment
    # E2 instrumental variable
    
    
    E1=[]
    E2=[]

    parametros=0;

    for i in range (0,n):
        E1.append([])
        E2.append([])
        for j in range (0,n):
            if len(C[i][j])>0:
                E1[i].append( filtra(C[i][j],e1[:,j:j+1]) )
                E2[i].append( filtra(C[i][j],e2[:,j:j+1]) )
                parametros=parametros+len(C[i][j]);
            else:
                E1[i].append( np.empty(shape=(0,0)) )
                E2[i].append( np.empty(shape=(0,0)) )
                
    #print('Total de parametros',parametros)
    #print('E1',E1)
     
    #return E1
    total=0

    # Filter signals and make ZY matrices
    Z=np.zeros((parametros,parametros))
    Y=np.zeros((parametros,1))
    
    for i in range (0,n):    
        EE1=np.zeros((N,parametros))
        EE2=np.zeros((N,parametros))
        for j in range (0,n):
            if E1[i][j].shape[1]>0:
                par=E1[i][j].shape[1]
                EE1[:,total:total+par]=E1[i][j] # monta [phi_i(1) phi_i(2)... phi_i(N); 0 0 ... 0 ]^T
                EE2[:,total:total+par]=E2[i][j]
            else:
                par=0
            total=total+par
            print(total)
        Z=Z+np.dot(EE1.T,EE2)
        Y=Y+np.dot(EE1.T,u[:,i:i+1])


    #print('Matriz Z',Z)
    #print('Matriz Y',Y)

    # Compute controller parameters
    p=np.dot(np.linalg.inv(Z),Y)
    return p     
        
        
#p=vrft_mimo(u,y,y,T,C,T)

#print(p)



    