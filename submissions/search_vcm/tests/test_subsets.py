import unittest

from submissions.search_vcm.subsets import HARD3, HARD8, get_subset, validate_subset


class SubsetTest(unittest.TestCase):
    def test_fixed_subsets(self):
        self.assertEqual(HARD3, [59, 60, 62])
        self.assertEqual(HARD8, [59, 60, 62, 56, 57, 58, 61, 63])
        for name in ("smoke", "hard3", "hard8", "strat64", "full600"):
            subset = get_subset(name)
            validate_subset(subset)
        self.assertEqual(len(get_subset("strat64")), 64)
        self.assertEqual(len(get_subset("full600")), 600)

    def test_strat64_deterministic(self):
        self.assertEqual(get_subset("strat64"), get_subset("strat64"))


if __name__ == "__main__":
    unittest.main()

