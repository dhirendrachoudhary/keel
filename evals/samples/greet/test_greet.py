import unittest

from greet import greet


class GreetTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(greet("Keel"), "Hello, Keel!")

    def test_other_name(self):
        self.assertEqual(greet("Ada"), "Hello, Ada!")


if __name__ == "__main__":
    unittest.main()
