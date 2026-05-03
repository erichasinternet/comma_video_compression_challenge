import unittest

import torch

from submissions.search_vcm_v2.families.boundary_residual_codec import (
    apply_records,
    boundary_map,
    candidate_from_records,
    dilate_bool,
    make_tile_records,
    pack_records,
    unpack_records,
)


class BoundaryResidualCodecTest(unittest.TestCase):
    def test_boundary_map_marks_four_neighbors(self):
        classes = torch.zeros(1, 4, 4, dtype=torch.uint8)
        classes[0, 1:3, 1:3] = 2
        b = boundary_map(classes)
        self.assertTrue(b[0, 1, 1])
        self.assertTrue(b[0, 0, 1])
        self.assertFalse(b[0, 0, 0])

    def test_dilate_bool_expands_radius(self):
        mask = torch.zeros(1, 5, 5, dtype=torch.bool)
        mask[0, 2, 2] = True
        out = dilate_bool(mask, 1)
        self.assertEqual(int(out.sum()), 9)

    def test_tile_pack_unpack_apply_exact_local_decode(self):
        low = torch.zeros(1, 8, 8, dtype=torch.uint8)
        exact = low.clone()
        exact[0, 2, 3] = 4
        exact[0, 5, 6] = 2
        records = make_tile_records(exact, low, [(0, 0, 0)], tile_size=8)
        self.assertEqual(len(records), 1)
        streams = pack_records(records, shape=tuple(exact.shape), tile_size=8)
        decoded = unpack_records(streams)
        repaired = apply_records(low, decoded, tile_size=8)
        self.assertTrue(torch.equal(repaired, exact))

    def test_candidate_reports_compressed_bytes(self):
        low = torch.zeros(1, 8, 8, dtype=torch.uint8)
        exact = low.clone()
        exact[0, 0, 0] = 1
        records = make_tile_records(exact, low, [(0, 0, 0)], tile_size=8)
        candidate = candidate_from_records(name="small", records=records, shape=tuple(exact.shape), tile_size=8)
        self.assertGreater(candidate["residual_bytes"], 0)
        self.assertEqual(candidate["source_error_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
