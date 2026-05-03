import unittest

from submissions.search_vcm.candidate_api import Budget
from submissions.search_vcm.asha import apply_promotion
from submissions.search_vcm.evaluator import base_metrics
from submissions.search_vcm.families.posenet_preprocess_oracle import PoseNetPreprocessOracle
from submissions.search_vcm.families.q55_pareto_controls import Q55ParetoControlCandidate
from submissions.search_vcm.families.q55_fallback_packaging import Q55FallbackCandidate


class CandidateSmokeTest(unittest.TestCase):
    def test_fallback_role(self):
        ctx = {"run_id": "test", "base": base_metrics(), "subset_indices": [0, 1]}
        row = Q55FallbackCandidate("q55_fp16_pose_int10").decision_row(Budget(round="smoke", subset="smoke"), ctx)
        self.assertEqual(row.role, "fallback_candidate")
        self.assertTrue(row.packable)
        self.assertEqual(row.decision, "fallback_recorded")

    def test_b1_oracle_nonpackable(self):
        ctx = {"run_id": "test", "base": base_metrics(), "subset_indices": [59, 60, 62]}
        row = PoseNetPreprocessOracle("b1_direct_preprocess").decision_row(Budget(round="hard8", subset="hard8"), ctx)
        self.assertEqual(row.kind, "oracle_only")
        self.assertFalse(row.packable)
        self.assertIsNone(row.archive_bytes)
        self.assertEqual(row.decision, "diagnostic_only")

    def test_pareto_registration_does_not_promote_without_optimizer(self):
        ctx = {"run_id": "test", "base": base_metrics(), "subset_indices": [59, 60, 62]}
        candidate = Q55ParetoControlCandidate(
            "c3_test",
            {"control": "c3", "variant": "shared_basis"},
            "shared_basis_test",
            added_bytes=10_000,
        )
        row = candidate.decision_row(Budget(round="hard8", subset="hard8"), ctx)
        row = apply_promotion(row, round_name="hard8")
        self.assertEqual(row.decision, "registered_not_promoted")
        self.assertIn("requires optimizer", row.failure_reason)


if __name__ == "__main__":
    unittest.main()
