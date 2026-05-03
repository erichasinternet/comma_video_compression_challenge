import unittest
from pathlib import Path

from submissions.search_vcm_v2.subsets import HARD3, HARD8, get_subset, strat64_from_qpose14, validate_subset


class SubsetTest(unittest.TestCase):
    def test_fixed_hard_subsets(self):
        self.assertEqual(HARD3, [59, 60, 62])
        self.assertEqual(HARD8, [59, 60, 62, 56, 57, 58, 61, 63])
        validate_subset(HARD3)
        validate_subset(HARD8)

    def test_strat64_is_deterministic_without_ledger(self):
        a = strat64_from_qpose14(ledger_path=Path("/tmp/missing-qpose-ledger.jsonl"))
        b = strat64_from_qpose14(ledger_path=Path("/tmp/missing-qpose-ledger.jsonl"))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        validate_subset(a)

    def test_round_aliases(self):
        self.assertEqual(get_subset("hard8_capacity"), HARD8)
        self.assertEqual(get_subset("hard8_compressed"), HARD8)
        self.assertEqual(get_subset("packability"), HARD8)

    def test_validate_rejects_bad_subset(self):
        with self.assertRaises(ValueError):
            validate_subset([1, 1])
        with self.assertRaises(ValueError):
            validate_subset([600])


if __name__ == "__main__":
    unittest.main()
