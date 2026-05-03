import unittest

import torch

from submissions.search_vcm_v2.families.lowmask_renderer import build_lowmask_renderer
from submissions.search_vcm_v2.families.pack_lowmask_renderer import estimate_lowmask_renderer_bytes


class LowmaskRendererTest(unittest.TestCase):
    def test_forward_capacity(self):
        model = build_lowmask_renderer("capacity")
        mask = torch.randint(0, 5, (2, 32, 48), dtype=torch.long)
        pose = torch.zeros(2, 6)
        z = torch.zeros(2, model.z_pose_dim)
        with torch.no_grad():
            f1, f2 = model(mask, pose, z)
        self.assertEqual(tuple(f1.shape), (2, 3, 32, 48))
        self.assertEqual(tuple(f2.shape), (2, 3, 32, 48))

    def test_byte_estimates_exist(self):
        for name in ("L48", "L40", "L32"):
            estimate = estimate_lowmask_renderer_bytes(name)
            self.assertGreater(estimate["params"], 0)
            self.assertGreater(estimate["int8_brotli_bytes_random_init"], 0)


if __name__ == "__main__":
    unittest.main()
