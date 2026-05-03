import tempfile
import unittest
from pathlib import Path

from submissions.search_vcm_v2.ledger import SOURCE_Q55_V1_LEDGER, build_proxy_qpose14_rows, qpose_subset_summary
from submissions.search_vcm_v2.subsets import HARD8


class QPose14BaselineTest(unittest.TestCase):
    def test_proxy_rows_match_reference_shape(self):
        if not SOURCE_Q55_V1_LEDGER.exists():
            self.skipTest("v1 q55 proxy ledger is not present")
        rows = build_proxy_qpose14_rows(SOURCE_Q55_V1_LEDGER)
        self.assertEqual(len(rows), 600)
        self.assertIn("qpose14_quality", rows[0])
        self.assertIn("rank_by_quality", rows[0])
        summary = qpose_subset_summary(rows, HARD8, archive_bytes=287573)
        self.assertEqual(summary["sample_count"], 8)
        self.assertGreater(summary["quality"], 0.0)


if __name__ == "__main__":
    unittest.main()
