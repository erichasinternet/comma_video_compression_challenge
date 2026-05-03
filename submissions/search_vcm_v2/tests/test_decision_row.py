import unittest

from submissions.search_vcm_v2.candidate_api import DecisionRow
from submissions.search_vcm_v2.evaluator import add_baseline_deltas, qpose14_reference_summary


class DecisionRowTest(unittest.TestCase):
    def test_baseline_relative_fields(self):
        base = qpose14_reference_summary()
        row = DecisionRow(
            run_id="r",
            candidate_name="c",
            family="factorized_exactmask_pose_tokens",
            role="exploratory_candidate",
            kind="packable_candidate",
            packable=True,
            config_hash="h",
            novelty_reason="",
            subset="hard8",
            round="hard8_capacity",
            archive_bytes=250000,
            added_bytes=0,
            quality=0.12,
            segnet_dist=0.0005,
            posenet_dist=0.00049,
            score=0.2865,
            seg_delta=None,
            pose_delta=None,
            byte_delta=-37573,
            score_delta_vs_base=-0.038,
            dominates_base=True,
            term_tradeoff="rate_quality_tradeoff_with_positive_net_score",
            decision="promote",
            failure_reason="",
        ).to_dict()
        out = add_baseline_deltas(row, base)
        self.assertEqual(out["baseline_name"], "qpose14")
        self.assertLess(out["rate_delta_vs_baseline"], 0.0)
        self.assertLess(out["estimated_score_delta_vs_baseline"], 0.0)


if __name__ == "__main__":
    unittest.main()
