import unittest

from submissions.search_vcm.evaluator import normalize_q55_metrics, quality, required_quality_for_score, score


class ScoreMathTest(unittest.TestCase):
    def test_quality_and_score_match_q55_int10(self):
        metrics = normalize_q55_metrics("q55_fp16_pose_int10")
        self.assertAlmostEqual(quality(metrics["segnet_dist"], metrics["posenet_dist"]), metrics["quality"], places=10)
        self.assertAlmostEqual(score(metrics["segnet_dist"], metrics["posenet_dist"], metrics["archive_bytes"]), metrics["score"], places=10)
        self.assertAlmostEqual(required_quality_for_score(288268), 0.1080541713013779, places=10)


if __name__ == "__main__":
    unittest.main()

