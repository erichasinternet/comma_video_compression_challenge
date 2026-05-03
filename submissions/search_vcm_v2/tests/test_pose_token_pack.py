import unittest

import torch

from submissions.search_vcm_v2.families.pack_pose_tokens import estimate_pose_token_bytes, pack_pose_tokens, quantize_pose_tokens, unpack_pose_tokens


class PoseTokenPackTest(unittest.TestCase):
    def test_quantize_roundtrip_shape(self):
        tokens = torch.randn(8, 16)
        q, meta = quantize_pose_tokens(tokens)
        self.assertEqual(tuple(q.shape), tuple(tokens.shape))
        self.assertEqual(meta["shape"], [8, 16])

    def test_pack_roundtrip(self):
        tokens = torch.randn(12, 24)
        restored = unpack_pose_tokens(pack_pose_tokens(tokens))
        self.assertEqual(tuple(restored.shape), tuple(tokens.shape))
        max_err = (tokens - restored).abs().max().item()
        self.assertLess(max_err, 0.05)

    def test_byte_estimate(self):
        estimate = estimate_pose_token_bytes(600, 24)
        self.assertEqual(estimate["raw_int8_bytes"], 600 * 24)
        self.assertGreater(estimate["packed_zero_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
