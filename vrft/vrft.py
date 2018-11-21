#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:43:36 2018

@author: diego
"""

from scipy import signal
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def pid(u, y, ad, bd, modo,P):

    N = np.size(y)

    # Filtro Td : ef = Td (Td^-1 - 1)y= (1-Td)y
    # Filtro Td : uf = Td u

    yf = signal.lfilter(bd, ad, y)
    uf = signal.lfilter(bd, ad, u)
    ef=y-yf

    # Filtro S=(1-Td)
    uf = signal.lfilter(ad-bd, ad, uf)
    ef = signal.lfilter(ad-bd, ad, ef)

    # Proporcional
    a0=np.array([1, 0,0])
    b0=np.array([1, 0,0])
    e_1 = signal.lfilter(b0, a0, ef)

    # Integral
    a0=np.array([1, -1,0])
    b0=np.array([1, 0,0])
    e_2 = signal.lfilter(b0, a0, ef)

    # Derivativo
    a0=np.array([1, 0,0])
    b0=np.array([1, -1,0])
    e_3 = signal.lfilter(b0, a0, ef)

    # Loco
    if modo=='loco':
        b=np.roots(bd)
        a0=np.array([1, -1])
        b0=np.array([1, -b])
        e_1 = signal.lfilter(b0, a0, ef)

    M=1
    uf=uf[M:N-M]    
    e_1=e_1[M:N-M]    
    e_2=e_2[M:N-M]    
    e_3=e_3[M:N-M]    


    if modo=='pid':
        phi = np.array([e_1,e_2,e_3])
    elif modo=='pd':
        phi = np.array([e_1,e_3])
    elif modo=='pi':
        phi = np.array([e_1,e_2])
    elif modo=='p':
        phi = np.array([e_1])
    elif modo=='loco':
        phi = np.array([e_1])
        

    R = np.dot(phi, phi.T)
    S = np.dot(phi, uf)
    rho = la.solve(R, S)
    
    print("- VRFT1:")

    if modo=='pid':
        kp = rho[0]
        ki = rho[1]
        kd = rho[2]
        print("PID(z):",kp,ki,kd)
        print("Zeros:",np.roots(np.array([kp+ki+kd, -(kp+2*kd),kd])))
        print("PID(s):",kp,ki*P,kd/P)

    elif modo=='pd':
        kp = rho[0]
        kd = rho[1]
        print("PD(z):",kp,kd)
        print("Zero:",kd/(kp+kd))
        print("PD(s):",kp,kd/P)
    elif modo=='pi':
        kp = rho[0]
        ki = rho[1]
        print("PI(z):",kp,ki)
        print("Zero:",kp/(kp+ki))
        print("PI(s):",kp,ki*P)
    elif modo=='p':
        kp = rho[0]
        print("P:",kp)
    elif modo=='loco':
        kp = rho[0]*b[0]
        ki = rho[0]*(1-b[0])
        print("PI(z):",kp,ki)
        print("Zero:",kp/(kp+ki))
        print("PI(s):",kp,ki*P)

    return rho


def pid_mf(u, y, ad, bd, modo,P,ti,tf,c=0.99):

    N = np.size(y)

    # Filtro Td : ef = Td (Td^-1 - 1)y= (1-Td)y
    # Filtro Td : uf = Td u

    yf = signal.lfilter(bd, ad, y)
    uf = signal.lfilter(bd, ad, u)
    ef=y-yf

    # Filtro C^-1
    b0=np.array([1, -1])
    a0=np.array([1, -c])
    uf = signal.lfilter(b0, a0, uf)
    ef = signal.lfilter(b0, a0, ef)

    # Proporcional
    b0=np.array([1, 0, 0])
    a0=np.array([1, 0, 0])
    e_1 = signal.lfilter(b0, a0, ef)

    # Integral
    b0=np.array([1,  0, 0])
    a0=np.array([1, -1, 0])
    e_2 = signal.lfilter(b0, a0, ef)

    # Derivativo
    b0=np.array([1, -1, 0])
    a0=np.array([1, 0, 0])
    e_3 = signal.lfilter(b0, a0, ef)

    # Loco
    if modo=='loco':
        b=np.roots(bd)
        a0=np.array([1, -1])
        b0=np.array([1, -b])
        e_1 = signal.lfilter(b0, a0, ef)

    Mi=250*ti
    Mf=250*tf
    uf=uf[Mi:N-Mf]    
    e_1=e_1[Mi:N-Mf]    
    e_2=e_2[Mi:N-Mf]    
    e_3=e_3[Mi:N-Mf]    

    if modo=='pid':
        phi = np.array([e_1,e_2,e_3])
    elif modo=='pd':
        phi = np.array([e_1,e_3])
    elif modo=='pi':
        phi = np.array([e_1,e_2])
    elif modo=='p':
        phi = np.array([e_1])
    elif modo=='loco':
        phi = np.array([e_1])
        



    R = np.dot(phi, phi.T)
    S = np.dot(phi, uf)
        
    rho = la.solve(R, S)

    #print("- VRFT:")

    if modo=='pid':
        kp = rho[0]
        ki = rho[1]
        kd = rho[2]
        print("PID(z):",kp,ki,kd)
        print("Zeros:",np.roots(np.array([kp+ki+kd, -(kp+2*kd),kd])))
        print("PID(s):",kp,ki*P,kd/P)

    elif modo=='pd':
        kp = rho[0]
        kd = rho[1]
        print("PD(z):",kp,kd)
        print("Zero:",kd/(kp+kd))
        print("PD(s):",kp,kd/P)
    elif modo=='pi':
        kp = rho[0]
        ki = rho[1]
        print("PI(z):",kp,ki)
        print("Zero:",kp/(kp+ki))
        print("PI(s):",kp,ki*P)
    elif modo=='p':
        kp = rho[0]
        print("New PID(s):",kp)
    elif modo=='loco':
        kp = rho[0]*b[0]
        ki = rho[0]*(1-b[0])
        #print("PI(z):",kp,ki)
        #print("Zero:",kp/(kp+ki))
        print("New PID(s):",kp,ki*P,0)

    return rho


def oci_arx(u, y, a):

    [theta]=sysid.arx(2, 1, u, y, 1)
    kd=theta[1]/theta[2]*(1-a);
    kp=-theta[0]/theta[2]*(1-a)-2*kd;
    ki=1/theta[2]*(1-a)-kd-kp;

    print("PID(z):",kp,ki,kd)
    print("Zeros:",np.roots(np.array([kp+ki+kd, -(kp+2*kd),kd])))
    print("PID(s):",kp,ki*250,kd/250)

    return [kp, ki, kd]

def oci_armax(u, y, a):

    [theta]=sysid.armax(2, 1, 1, u, y, 1,1000)

    kd=theta[1]/theta[2]*(1-a);
    kp=-theta[0]/theta[2]*(1-a)-2*kd;
    ki=1/theta[2]*(1-a)-kd-kp;

    print("PID(z):",kp,ki,kd)
    print("Zeros:",np.roots(np.array([kp+ki+kd, -(kp+2*kd),kd])))
    print("PID(s):",kp,ki*250,kd/250)

    return [kp, ki, kd]
