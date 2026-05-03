import tempfile
import unittest
from pathlib import Path

from submissions.search_vcm.evaluator_validation import (
    compare_reference,
    parse_report,
    quality,
    score,
)


REPORT_TEXT = """=== Evaluation config ===
  batch_size: 16
  device: mps
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00065135
  Average SegNet Distortion: 0.00072222
  Submission file size: 288,268 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00767783
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.34
"""


class EvaluatorValidationTest(unittest.TestCase):
    def test_parse_report_computes_full_precision_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.txt"
            report.write_text(REPORT_TEXT)
            got = parse_report(report)
        self.assertEqual(got["samples"], 600)
        self.assertEqual(got["archive_bytes"], 288268)
        self.assertAlmostEqual(got["quality"], quality(0.00072222, 0.00065135), places=12)
        self.assertAlmostEqual(
            got["score_full_precision"],
            score(0.00072222, 0.00065135, 288268, 37545489),
            places=12,
        )

    def test_reference_compare_passes_q55_int10_report_values(self):
        metrics = {
            "archive_bytes": 288268,
            "posenet_dist": 0.00065135,
            "segnet_dist": 0.00072222,
        }
        got = compare_reference("q55_fp16_pose_int10", metrics)
        self.assertTrue(got["available"])
        self.assertTrue(got["pass"])


if __name__ == "__main__":
    unittest.main()
