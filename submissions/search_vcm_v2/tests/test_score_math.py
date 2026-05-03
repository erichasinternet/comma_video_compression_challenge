import unittest

from submissions.search_vcm_v2.evaluator import ORIGINAL_BYTES, qpose14_reference_summary, quality, rate_term, required_quality_for_score, score


class ScoreMathTest(unittest.TestCase):
    def test_qpose14_reference_score(self):
        summary = qpose14_reference_summary()
        self.assertAlmostEqual(summary["quality"], quality(0.00061261, 0.00052154), places=10)
        self.assertAlmostEqual(summary["rate_term"], 25.0 * 287573 / ORIGINAL_BYTES, places=10)
        self.assertAlmostEqual(summary["score"], score(0.00061261, 0.00052154, 287573), places=10)
        self.assertLess(summary["score"], 0.326)
        self.assertGreater(summary["score"], 0.324)

    def test_required_quality_targets(self):
        self.assertAlmostEqual(required_quality_for_score(260000), 0.12688, places=4)
        self.assertAlmostEqual(required_quality_for_score(250000), 0.13354, places=4)
        self.assertAlmostEqual(required_quality_for_score(240000), 0.14021, places=4)

    def test_rate_monotonic(self):
        self.assertLess(rate_term(240000), rate_term(250000))


if __name__ == "__main__":
    unittest.main()
