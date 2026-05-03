import unittest

from submissions.search_vcm_v2.families.lowmask_data import FP4_ARCHIVE, archive_audit


class LowmaskDataTest(unittest.TestCase):
    def test_archive_audit_when_present(self):
        if not FP4_ARCHIVE.exists():
            self.skipTest("fp4_mask_gen archive not present")
        audit = archive_audit(FP4_ARCHIVE)
        self.assertEqual(audit["archive_bytes"], FP4_ARCHIVE.stat().st_size)
        self.assertIn("mask.obu.br", audit["payload_breakdown"])
        self.assertIn("model.pt.br", audit["payload_breakdown"])
        self.assertIn("pose.bin.br", audit["payload_breakdown"])


if __name__ == "__main__":
    unittest.main()
