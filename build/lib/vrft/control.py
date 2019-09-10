#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:45:33 2018
@authors: Diego Eckhard and Emerson Boeira
"""
#%% Header: import libraries

from scipy import signal  # signal processing library
import numpy as np  # important package for scientific computing
import vrft  # vrft package

#%% Function that is used to filter in a MIMO structure


def filter(G, u):
    # Description to help the user
    """Function used to filter the signals in a MIMO structure.
    Inputs: G,u
    Outputs: y
    
    Inputs description:
        G: Transfer matrix of the MIMO filter. It's a python list of TransferFunctionDiscrete elements. The dimension of the transfer matrix list must be (n,m), where n=number of outputs and m=number of inputs;
        u: Input data matrix. The dimension of u must be (N,m), where N is the data length and m is the number of inputs of the system.
            
    Outputs description:
        y: Output data matrix. The dimension of y is (N,n), where N is the data length and n is the number of outputs of the system."""

    # testing the type of G set by the user and converting it to list
    if isinstance(G, signal.ltisys.TransferFunctionDiscrete):
        G = [[G]]

    # number of outputs
    n = len(G)
    # number of inputs
    m = len(G[0])
    # preallocating the output matrix
    y = np.zeros((len(u), n))
    # loop to calculate each output signal
    for i in range(0, n):
        for j in range(0, m):
            if G[i][j] != 0:
                t, v = signal.dlsim(G[i][j], u[:, j])
                y[:, i] = y[:, i] + v[:, 0]
    # return the output (filtered) signal
    return y


#%% Function that is used to filter every column of a matrix with one filter


def colfilter(G, u):
    # Function description to help the user
    """Function that filters every column of a matrix with the same filter.    
    Inputs: G,u
    Outputs: y
        
    Inputs description:
        G: SISO filter. It must be a TransferFunctionDiscrete element;
        u: Matrix with several columns that will be filtered by G. The dimension of u must be (N,x) where N is the data length and x is the number of columns that will be filtered.
    
    Outputs description:
        y: filtered data matrix. The dimension of y is (N,x)."""

    # preallocating the output matrix
    y = np.zeros((np.shape(u)))
    # test if the transfer function is not zero
    if G != 0:
        # loop to filter each column of the matrix u
        for i, col in enumerate(u.T):
            t, v = signal.dlsim(G, col)
            y[:, i] = v[:, 0]
    return y


#%% Function that design the controller with the VRFT method


def design(u, y, y_iv, Td, C, L):
    # Description of the design function to help the user
    """Function that design the controller using the VRFT method.
    Inputs: u,y,y_iv,Td,C,L
    Output: p
    
    Inputs description:
        u: input data matrix. The dimension of u must be (N,n), where N is the data length and n is the number of inputs/outputs of the system;
        y: output data matrix. The dimension of y must be (N,n), where N is the data length and n is the number of inputs/outputs of the system;
        y_iv: output data matrix for the instrumental variable. The dimension of y must also be (N,n). If the user doesn't have any instrumental variable data, then y_iv must be the same data matrix as y;
        Td: Reference Model transfer matrix. It must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n);
        C: Controller structure that will be used on the method. It also must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n);
        L: VRFT method filter. It also must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n).
        
    Outputs description:
        p: controller parameters obtained by the VRFT method.
        The parameter vector p is organized as p=[p11^T p12^T ... p1n^T p21^T p22^T ... p2n^T ... pnn^T]^T.
        Each pij represents the parameter vector of each subcontroller Cij(z,pij)."""

    # Tests for the SISO scenario:
    # testing the type of Td set by the user and converting it to list
    if isinstance(Td, signal.ltisys.TransferFunctionDiscrete):
        Td = [[Td]]
    # testing the type of L set by the user and converting it to list
    if isinstance(L, signal.ltisys.TransferFunctionDiscrete):
        L = [[L]]
    # testing the type of L set by the user and converting it to list
    if isinstance(C[0][0], signal.ltisys.TransferFunctionDiscrete):
        C = [[C]]

    # number of data samples/ data length
    N = len(u)
    # number of inputs/outputs of the system
    n = len(Td)
    # creates a dummy time vector, necessary for the vrft.stbinv function
    t = np.linspace(0, N - 1, N)  # linspace(start,stop,numberofpoints)
    # pushing the vector to have the specified dimensions
    t.shape = (1, N)

    # Filter the signal u
    uf = vrft.filter(L, u)

    # transformation of Td from the MIMO transfer function list structure to a state-space model
    Atd, Btd, Ctd, Dtd = vrft.mtf2ss(Td)
    # calculates the virtual reference for the first data set
    rv, _, flagvr = vrft.stbinv(Atd, Btd, Ctd, Dtd, y.T, t)
    rv = rv.T
    # calculates the virtual reference for the second data set (instrumental variable)
    rv_iv, _, _ = vrft.stbinv(Atd, Btd, Ctd, Dtd, y_iv.T, t)
    rv_iv = rv_iv.T

    # test if the inversion algorithm was succesful
    if flagvr == 0:
        # if flagvr=0, then, the inversion algorithm was succesful
        # remove the last samples of y, to match the dimensions of the virtual reference
        # number of samples used in the method
        N = rv.shape[0]
        y = y[0:N, :]
        y_iv = y_iv[0:N, :]
        # virtual error
        ebar = rv - y
        ebar_iv = rv_iv - y_iv
        # remove the last samples of the input (to match the dimension of the virtual error)
        uf = uf[0:N, :]

        # calculating the number of parameters of each controller and organizing it as a matrix
        # preallocating
        nbpar = np.zeros((n, n))
        # loop
        for i in range(0, n):
            for j in range(0, n):
                nbpar[i][j] = len(C[i][j])
        # total number of parameters (casting to integer)
        p_tot = int(np.sum(nbpar))

        # assembling the matrices phi_iN and organizing it as a python list
        # initializing the list
        phi_iN_list = []
        csi_iN_list = []
        # loops
        for i in range(0, n):
            # preallocating the matrices
            phi_iN = np.empty((N, 0))
            csi_iN = np.empty((N, 0))
            # loop on j
            for j in range(0, n):
                if len(C[i][j]) > 0:
                    # calculating phi_ijN^T
                    phi_ijN = vrft.filter(C[i][j], ebar[:, j : j + 1])
                    # calculating csi_ijN^T (instrumental variable)
                    csi_ijN = vrft.filter(C[i][j], ebar_iv[:, j : j + 1])
                    # calculating phi_iN^T, by concatenating the phi_ijN^T matrices
                    phi_iN = np.concatenate(
                        (phi_iN, phi_ijN), axis=1
                    )  # concatenate column wise
                    # instrumental variable
                    csi_iN = np.concatenate(
                        (csi_iN, csi_ijN), axis=1
                    )  # concatenate column wise
            # saving in the list structure
            phi_iN_list.append(phi_iN)
            csi_iN_list.append(csi_iN)

        # assembling the matrices Phi_vrf and Csi_vrf (instrumental variable) - which considers the filter L of the VRFT method
        # initializing the Phi_vrf and Csi_vrf matrices
        Phi_vrf = np.empty((0, p_tot))
        Csi_vrf = np.empty((0, p_tot))
        # start the loop
        # on i
        for i in range(0, n):
            # initializing the matrices that compososes "each row" of Phi_vrf and Csi_vrf
            Phi_row = np.empty((N, 0))
            Csi_row = np.empty((N, 0))
            # loop on j
            for j in range(0, n):
                Phi_ij = colfilter(L[i][j], phi_iN_list[j])
                Csi_ij = colfilter(L[i][j], csi_iN_list[j])
                # concatenating the columns to assemble "each row" of Phi_vrf and Csi_vrf
                Phi_row = np.concatenate(
                    (Phi_row, Phi_ij), axis=1
                )  # concatenate column wise
                Csi_row = np.concatenate(
                    (Csi_row, Csi_ij), axis=1
                )  # concatenate column wise
            # concatanating the rows of Phi_vrf and Csi_vrf
            Phi_vrf = np.concatenate((Phi_vrf, Phi_row), axis=0)  # concatenate row wise
            Csi_vrf = np.concatenate((Csi_vrf, Csi_row), axis=0)  # concatenate row wise

        # reorganizing the uf vector (stacking)
        # preallocating
        Uf = np.empty((0, 1))
        # loop
        for i in range(0, n):
            Uf = np.concatenate((Uf, uf[:, i : i + 1]), axis=0)  # concatenate row wise

        # compute controller parameters
        Z = np.matmul(Csi_vrf.T, Phi_vrf)
        Y = np.matmul(Csi_vrf.T, Uf)
        p = np.linalg.solve(Z, Y)

        # returning the parameter vector
        return p

    elif flagvr == 1:
        # if flagvr=1, then it was not possible to calculate the inverse of the reference model
        print(
            "It was not possible to calculate the virtual reference. The inversion algorithm has failed."
        )
        # return an empty parameter vector
        p = np.empty((0, 0))
        return p

    elif flagvr == 2:
        # if flagvr=2, the inverse of the reference model is unstable. VRFT method aborted!
        print(
            "The inverse of the reference model Td(z) is unstable. It is not recommended to proceed with the VRFT method. The algorithm was aborted!"
        )
        # return an empty parameter vector
        p = np.empty((0, 0))
        return p
