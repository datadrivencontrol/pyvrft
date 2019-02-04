import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from vrft import vrft

N=1000

# Input signal
u=np.zeros((N,1))
u[2]=1
u[202]=1
u[402]=1
u[602]=1
u[802]=1

# System definition
a0=np.array([1.00, -1.70, 0.72])
b0=np.array([0.00,  0.04, 0.00])
c0=np.array([1.00, -0.30, 0.50])

# Output signal
y = signal.lfilter(b0, a0, u)
e = np.random.randn(N)/1000
v = signal.lfilter(c0, a0, e)
y=y+e

# Reference model
a=0.9
a0=np.array([1, -a])
b0=np.array([0, 1-a])


# Controller
rho_vrft=vrft.pid(u, y,a0,b0,'pid',250)

# Plot
p1=plt.figure(1)
plt.plot(y, label='y')
plt.xlabel('sample')
plt.legend()

