import unittest
import vrft
import scipy.signal as signal
import numpy as np 

class TestVrft(unittest.TestCase):
    def test_tf2ss(self):
        
        
        G11 = signal.TransferFunction([1],[1,-0.9],dt=1)
        G12 = 0
        G21 = 0
        G22 = signal.TransferFunction([1],[1,-0.9],dt=1)
        G = [[G11,G12],[G21,G22]]
        Td11 = signal.TransferFunction([0.2],[1,-0.8],dt=1)
        Td12 = 0
        Td21 = 0
        Td22 = signal.TransferFunction([0.2],[1,-0.8],dt=1)
        Td = [[Td11,Td12],[Td21,Td22]]
        L = Td
        Cpi = [[signal.TransferFunction([1, 0],[1, -1],dt=1)] , [signal.TransferFunction([1],[1, -1],dt=1)]] 
        C = [[Cpi,[]],[[],Cpi]] 
        N = 350
        t = np.linspace(0, N-1, N) 
        t.shape = (1,N)
        ts = N
        fs = 1/ts
        u1 = 0.5-0.5*signal.square(2*np.pi*fs*t).T
        u2 = 0.5-0.5*signal.square(2*np.pi*fs*t-3*np.pi/2).T
        u = np.concatenate((u1,u2),axis=1)
        y = vrft.filter(G,u)
        p = vrft.design(u,y,y,Td,C,L)
        
        p0 = np.array([[0.2],[-0.18],[ 0.2 ],[-0.18]])
        
        
        self.assertTrue( np.linalg.norm(p - p0) < np.finfo(np.float32).eps )

if __name__ == '__main__':
	unittest.main()

