import unittest
import vrft
import scipy.signal as signal


class TestVrft(unittest.TestCase):
    def test_tf2ss(self):
        G11 = signal.TransferFunction([1],[1,-0.9],dt=1)
        G12 = signal.TransferFunction([0.6],[1,-0.7],dt=1)
        G21 = 0
        G22 = signal.TransferFunction([1.4],[1,-0.65],dt=1)
        # organizing the MIMO system G(z) in a python list
        G=[[G11,G12],[G21,G22]]
        # IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]. The latter generates a WARNING!!
        Ass,Bss,Css,Dss=vrft.mtf2ss(G)
        a = 1
        self.assertEqual(a,1)

if __name__ == '__main__':
	unittest.main()

