import unittest

import torch

from submissions.search_vcm_v2.families.factorized_renderer import build_renderer, mask_boundary_map
from submissions.search_vcm_v2.families.pack_factorized_renderer import estimate_renderer_bytes


class FactorizedRendererTest(unittest.TestCase):
    def test_forward_shapes(self):
        model = build_renderer("F16")
        mask = torch.randint(0, 5, (2, 24, 32), dtype=torch.long)
        pose6 = torch.zeros(2, 6)
        z_pose = torch.zeros(2, model.z_pose_dim)
        frame1, frame2 = model(mask, pose6, z_pose)
        self.assertEqual(tuple(frame1.shape), (2, 3, 24, 32))
        self.assertEqual(tuple(frame2.shape), (2, 3, 24, 32))
        self.assertTrue(torch.isfinite(frame1).all())
        self.assertTrue(torch.isfinite(frame2).all())

    def test_boundary_map_marks_class_edges(self):
        mask = torch.zeros(1, 4, 4, dtype=torch.long)
        mask[:, :, 2:] = 1
        boundary = mask_boundary_map(mask)
        self.assertEqual(tuple(boundary.shape), (1, 1, 4, 4))
        self.assertGreater(float(boundary.sum()), 0.0)
        self.assertEqual(float(boundary[:, :, :, 0].sum()), 0.0)

    def test_byte_estimates_exist(self):
        data = estimate_renderer_bytes("F16")
        self.assertEqual(data["config_name"], "F16")
        self.assertGreater(data["params"], 0)
        self.assertGreater(data["int8_brotli_bytes_random_init"], 0)


if __name__ == "__main__":
    unittest.main()
