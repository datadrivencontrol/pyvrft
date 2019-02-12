import unittest
import vrft

class TestVrft(unittest.TestCase):
	def test_simple_vrft(self):
		a = vrft.design(1)
		self.assertEqual(a,1)

if __name__ == '__main__':
	unittest.main()

