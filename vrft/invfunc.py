# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:38:10 2019
@author: Emerson Boeira and Diego Eckhard
"""
"""
Functions that are used to implement the algorithm proposed on the paper: Stable Inversion of Linear Systems
"""
#%% Header: import python libraries

import numpy as np # important package for scientific computing
import scipy.linalg as scilin # importing linear algebra functions from scipy
import scipy.signal as signal # signal processing library

#%% Function that does the reduction of the system

def invredc(A,B,C,D,y,v):
        
    # calculate the number of samples of the output
    N=np.shape(y)[1] # the number of samples is the number of columns of y (we defined that way)
    
    # calculate systems dimensions: number of states, number of inputs and number of outputs
    n=A.shape[0] #number of states
    #m=B.shape[1] #number of inputs, maybe it's not necessary
    p=C.shape[0] #number of outputs

    # A. Output Basis Change
    # here the output basis change and its important quantities and matrices are calculated

    # rank of the feedforward matrix:
    r=np.linalg.matrix_rank(D)

    # to calculate the S1 matrx, we have partitioned the matrix into [S1a;S2a]
    # firstly, we obtain S1a
    # since D0 has to possess full row rank, rank(D0)=r, a simple way to do that is to use the scipy.linalg.orth function
    D0=(scilin.orth(D.transpose())).transpose()
    # calculating S1a as a solution of the problem S1a*D=D0 using the pseudoinverse (Moore-Penrose inverse):
    S1at=scilin.pinv(D.transpose())@D0.transpose()
    S1a=S1at.transpose()
    # S1b is the null space (kernel) of D from the left
    S1b=(scilin.null_space(D.transpose())).transpose()
    # assembling the S1 matrix
    S1=np.concatenate((S1a,S1b),axis=0) # axis=0 concatenate vertically (row wise)

    # the C2 matrix is obtained by a partition of S1*C, which can by also obtained with the use of S1b
    # calculating C2
    C2=S1b@C
    # rank of C2
    q=np.linalg.matrix_rank(C2)
    
    # calculating the matrix S2, which is very similar to S1, and it is also partitioned as S2=[S2a;S2b]
    # since C2bar has to possess full row rank, rank(C2)=q
    C2tilde=(scilin.orth(C2.transpose())).transpose()
    # calculating S2a as a solution of the problem S2a*C2=C2bar using the pseudoinverse (Moore-Penrose inverse):
    S2at=scilin.pinv(C2.transpose())@C2tilde.transpose()
    S2a=S2at.transpose()
    # S2b is the null space (kernel) of C2 from the left
    S2b=(scilin.null_space(C2.transpose())).transpose()
    # assembling the S2 matrix
    S2=np.concatenate((S2a,S2b),axis=0) # axis=0 concatenate vertically (row wise)
    
    # now that we have S1 and S2, we can assemble the S matrix
    # we defined the notation: S=Sa*S1, where Sa is partitioned as Sa=[I 0;0 S2]=[Sa1 Sa2]
    # partitions of Sa
    Sa11=np.identity(r)
    Sa12=np.zeros((r,p-r))
    Sa21=np.zeros((p-r,r))
    Sa22=S2
    # assembling the columns of Sa, Sa=[Sa1 Sa2]
    Sa1=np.concatenate((Sa11,Sa21),axis=0) # concatenate vertically (row wise)
    Sa2=np.concatenate((Sa12,Sa22),axis=0) # concatenate vertically (row wise)
    # finally, assembling the matrix Sa:
    Sa=np.concatenate((Sa1,Sa2),axis=1) # concatenate horizontally (column wise)
    # obtaining the S matrix by the multiplication
    S=Sa@S1
    
    # doing the transformation of the output ytilde=Sy
    ytilde=S@y
    # we'll not partition the output yet, first, we'll do the State-Space Basis Change
    
    # B. State-Space Basis Change
    # in this section we'll do the state-space basis change of the system

    # the first step is the calculation of the transformation matrix, as defined in the paper
    # we'll call T^{-1} as M, so C2tilde*M=[0 I]. And we'll partition M as M=[M1 M2]. C2tilde*M=[C2tilde*M1 C2tilde*M2]
    # since rank(C2tilde)=q, nullity(C2tilde)=n-q
    # M1 can be defined as a basis of the null space of C2tilde
    M1=scilin.null_space(C2tilde)
    # and M2 is the solution of the equation C2tilde*M2=I. To calculate this solution, we'll use the pseudoinverse again
    M2=scilin.pinv(C2tilde)
    # now, we assemble the M matrix with the concatenate function
    M=np.concatenate((M1,M2),axis=1) # concatenate horizontally (column wise)
    # finally, we calculate the T matrix by inverting M
    T=np.linalg.inv(M)
    
    # now, we proceed to the transformation of the state-space matrices
    # transformation of the system's dynamic matrix
    Atilde=T@A@M
    # transformation of the system's input matrix
    Btilde=T@B
    # transformation of the system's output matrix
    Ctilde=C@M
    # transformation of the system's feedforward matrix (it's the same)
    #Dtilde=D # actually, this step is not necessary
    # transformation of the aditional system input v
    vtilde=T@v
    
    # in the next step, we need to partition the new system's matrices and outputs
    
    # partition of the outputs
    # y1 has r lines and N columns
    y1=ytilde[0:r,:]
    # y2 has q lines and N columns, and it starts at the r+1 line (which in python is the r line since the vector index starts at 0)
    y2=ytilde[r:r+q,:]
    # y3 is irrelevant, then, it will be neglected
    
    # partitioning the system matrices
    # firstly, the system's dynamic matrix Atilde
    A11=Atilde[0:n-q,0:n-q]
    A12=Atilde[0:n-q,n-q:n]
    A21=Atilde[n-q:n,0:n-q]
    A22=Atilde[n-q:n,n-q:n]
    # the system's input matrix Btilde
    B1=Btilde[0:n-q,:]
    B2=Btilde[n-q:n,:]
    # the system's output matrix Ctilde
    C11=Ctilde[0:r,0:n-q]
    C12=Ctilde[0:r,n-q:n]

    #partition the aditional input vtilde
    v1=vtilde[0:n-q,:]
    v2=vtilde[n-q:n,:]

    # C. Reduction of State-Space Dimension
    # now, we'll do the reduction of the state-space system

    # following the equations in the paper
    # calculating y1hat
    y1hat=y1-C12@y2
    # we have to discard the last sample to make the dimensions of y1hat and y2hat match
    y1hat=y1hat[:,0:N-1]

    # calculating y2hat
    # preallocating variables before the loop
    y2hat=np.zeros((q,N-1))
    # runing the loop
    for k in range(0, N-1): # the for loop has to run N-1 times, from 0 to N-2, because of y2[k+1] on the equation
        y2hat[:,k]=y2[:,k+1]-A22@y2[:,k]-v2[:,k]
    
    # assembling the reduced system's output vector
    yhat=np.concatenate((y1hat,y2hat),axis=0)

    # calculating the aditional input vhat
    vhat=v1+A12@y2
    # discarding the last sample
    vhat=vhat[:,0:N-1]

    # now, we'll assemble the reduced state-space system
    # reduced system's dynamic matrix
    Ahat=A11
    # reduced system's input matrix
    Bhat=B1
    # reduced system's output matrix
    Chat=np.concatenate((C11,A21),axis=0) #concatenate vertically (row wise)
    # reduced system's feedforward matrix
    Dhat=np.concatenate((D0,B2),axis=0) #concatenate vertically (row wise)
    # calculating rhat, the new rank of the feedforward matrix Dhat - an important quantitie of the algorithm
    rhat=np.linalg.matrix_rank(Dhat)

    # calculating the new dimension of the reduced system
    # reduced system state vector dimension
    nhat=n-q
    # reduced system output vector dimension
    phat=r+q
    
    return Ahat,Bhat,Chat,Dhat,yhat,vhat,nhat,phat,rhat

#%% Function that does the whole inversion of the system. It uses the invreduction function defined above
    
def stbinv(A,B,C,D,y,t):
    
    # calculate the number of samples of the output
    N=np.shape(y)[1] # the number of samples is the number of columns of y (we defined that way)
    
    # calculate systems dimensions: number of states, number of inputs and number of outputs
    m=B.shape[1] #number of inputs
    # calculate systems dimensions: number of states, number of inputs and number of outputs
    n=A.shape[0] #number of states
    
    # initialize the variable v (aditional know input)
    v=np.zeros((n,N)) # it will be important later
    
    # initializing the flag variable
    flag=0
    # initializing the counter for the rounds of the reduction step of the algorithm
    kround=0
    
    # starting the loop of the reduction procedure
    while flag==0:
        # run a step of the reduction order algorithm     
        Ahat,Bhat,Chat,Dhat,yhat,vhat,nhat,phat,rhat=invredc(A,B,C,D,y,v)
        # increments the counter of reductions
        kround=kround+1 # increments the counter of the rounds        
                
        # preallocating the state vector of the inverse system
        xhat=np.zeros((nhat,N-kround)) # it must have N-kround samples
        # preallocating the calculated input
        uhat=np.zeros((m,N-kround))
                
        # defining the reduced time vector
        tt=t[:,0:N-kround]
        
        # test the conditions of invertibility
        if (phat < m):
            # if this condition is true, then the algorithm has failed and it is not possible to find the inverse
            flag=1
            # if this is the case, we print a message and end the execution
            print('The inversion algorithm has failed')
            return uhat,tt
        else:
            if (rhat==m):
                #((rhat==m)&(rhat==phat)):
                #if this condition is true, then the algorithm is done. We can invert the system
                flag=2                
                # calculating the inverse of the feedforward matrix
                #E=np.linalg.inv(Dhat)
                E=np.linalg.pinv(Dhat)                
            else:                
                # if none of the conditions above are true, then we need to proceed to another round of the reduction step of the algorithm
                A=Ahat;B=Bhat;C=Chat;D=Dhat;y=yhat;v=vhat          
                # after the reduction procedure is done, then the system can be inverted
                
    # calculating the dynamic matrix of the inverse system
    Ainv=Ahat-Bhat@E@Chat
    # eigenvalues of the inverse system's dynamic matrix
    wv, v = np.linalg.eig(Ainv) # w=eigenvalues, v=eigenvectors
    # calculating the input matrix of the inverse system
    Binv=Bhat@E
    # calculating the output matrix of the inverse system
    Cinv=-E@Chat
    # calculating the feedforward matrix of the inverse system
    Dinv=E

    # test if the inverse dynamic system is stable
    wbool=wv>1
    wsum=np.sum(wbool)
    # test if wsum is greater than 1
    if wsum>0:                    
        # if wsum is greater than 1, then, the inverse system is unstable, so we end the execution of the algorithm
        print('The inverse system is unstable')
        return uhat,tt                    
    else:                    
        # if wsum=0, then the inverse system is stable, and we can calculate the input signal   
        # calculate the first value for the output (t=0)
        uhat[:,0]=Cinv@xhat[:,0]+Dinv@yhat[:,0]
        # calculate the states and the output of the inverse system
        for k in range(0, N-1-kround):
            xhat[:,k+1]=Ainv@xhat[:,k]+Binv@yhat[:,k]+vhat[:,k]
            uhat[:,k+1]=Cinv@xhat[:,k+1]+Dinv@yhat[:,k+1]
                
    return uhat,tt

#%% Function that does the transformation of a MIMO transfer function process in a state-space model
# IMPORTANT: This is a simple algorithm that does not produce a minimal realization!

def mtf2ss(G):
    
    # calculating the number of outputs
    p=len(G)
    # calculating the number of inputs
    m=len(G[0])

    # creating a list for each matrix
    A=[];B=[];C=[];D=[]

    nss=0
    # loop that get the SISO state-space transformations
    for i in range(0, p):
        # outputs - first index of the MIMO process list
        A.append([]);B.append([]);C.append([]);D.append([])
        for j in range(0, m):
            # inputs - second index of the MIMO process list
            if (G[i][j]!=0):
                #transform the individual SISO systems to a state-space model
                Aij,Bij,Cij,Dij=signal.tf2ss(G[i][j].num, G[i][j].den)
                #calculate the size of the A matrix
                nss=nss+Aij.shape[0]
                #organizing the matrices iin a list
                A[i].append(Aij);B[i].append(Bij);C[i].append(Cij);D[i].append(Dij)
            else:
                A[i].append([]);B[i].append([]);C[i].append([]);D[i].append([])

    # preallocation of the system's matrix
    Ass=np.zeros((nss,nss))
    Bss=np.zeros((nss,m))
    Css=np.zeros((p,nss))
    Dss=np.zeros((p,m))
    # counters
    ct=0

    # loop that organize the MIMO list obtained above on the state-space model
    for i in range(0, p):    
        # loop on the outputs
        for j in range(0, m):
            # loop on the inputs
            #test if the matrix isn't zero
            if len(A[i][j])>0:
                # calculate the size of the dynamic matrix
                nij=A[i][j].shape[0]
                # organizing the dynamic matrix
                Ass[ct:ct+nij,ct:ct+nij]=A[i][j]
                # organizing the input matrix
                Bss[ct:ct+nij,j]=B[i][j][:,0]
                # organizing the output matrix
                Css[i,ct:ct+nij]=C[i][j][0,:]
                # organizing the feedforward matrix
                Dss[i,j]=D[i][j]
                # incremets the counter
                ct=ct+nij
                
    return Ass,Bss,Css,Dss