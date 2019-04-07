#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:31:27 2019
@authors: Diego Eckhard and Emerson Boeira
"""
#%% Header: importing python libraries

import numpy as np # important package for scientific computing

#%% Function that reads the data from a csv file

def datafromcsv(file_name,delim,row_offset,n):
    # Function  that reads the input and output data from a .csv to be used on the VRFT method
    # file_name: .csv file path
    # delim: column delimiter on the .csv file
    # row_offset: offset to be considered on the .csv file (since the first lines could be some kind of Header of the .csv file)
    # n: number of inputs and outputs of the system
    # IMPORTANT: each column of the .csv file must represent the following data
    # y1,y2,...,yn,u1,u2,...,un
    
    # read the data from a .csv using numpy
    x=np.genfromtxt(file_name,delimiter=delim,skip_header=row_offset)
    # separating inputs and outputs in different arrays
    y=x[:,0:n]
    u=x[:,n:n+n]
    
    # returning the data
    return y,u