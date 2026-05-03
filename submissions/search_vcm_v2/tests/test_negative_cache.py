import unittest

from submissions.search_vcm_v2.negative_cache import NegativeCache


class NegativeCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache = NegativeCache()

    def test_allows_live_family(self):
        decision = self.cache.check(family="qpose14_baseline")
        self.assertTrue(decision.allowed)

    def test_allows_lowmask_boundary_family(self):
        decision = self.cache.check(family="lowmask_boundary_residual")
        self.assertTrue(decision.allowed)

    def test_factorized_family_closed_after_gate1(self):
        decision = self.cache.check(family="factorized_exactmask_pose_tokens")
        self.assertFalse(decision.allowed)
        self.assertIn("dead_family", decision.reason)

    def test_lowmask_family_closed_after_gate1(self):
        decision = self.cache.check(family="lowmask_qpose_distill")
        self.assertFalse(decision.allowed)
        self.assertIn("dead_family", decision.reason)

    def test_blocks_dead_family(self):
        decision = self.cache.check(family="mask_range_av1")
        self.assertFalse(decision.allowed)
        self.assertIn("dead_family", decision.reason)

    def test_dead_family_requires_novelty_override(self):
        self.assertFalse(self.cache.check(family="mask_range_av1", allow_negative_cache=True).allowed)
        decision = self.cache.check(
            family="mask_range_av1",
            allow_negative_cache=True,
            novelty_reason="materially different coding target",
        )
        self.assertTrue(decision.allowed)

    def test_blocks_unknown_family(self):
        decision = self.cache.check(family="random_new_family")
        self.assertFalse(decision.allowed)
        self.assertIn("family_not_allowed", decision.reason)


if __name__ == "__main__":
    unittest.main()
