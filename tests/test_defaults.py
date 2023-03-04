import unittest

import ananke.defaults as defaults

class DefaultsTestCase(unittest.TestCase):
    def test_defaults(self):
        assert defaults.seed == 32118


if __name__ == '__main__':
    unittest.main()
