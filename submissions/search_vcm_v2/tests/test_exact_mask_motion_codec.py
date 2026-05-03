import unittest

import numpy as np

from submissions.search_vcm_v2.tools.exact_mask_motion_codec import (
    build_motion_records,
    compute_best_motion,
    decode_motion_streams,
    make_shifts,
    pack_motion_streams,
    pred_tile,
    shift_frame,
    zigzag_decode,
    zigzag_encode,
)


class ExactMaskMotionCodecTest(unittest.TestCase):
    def test_zigzag_roundtrip(self):
        for value in range(-32, 33):
            self.assertEqual(zigzag_decode(zigzag_encode(value)), value)

    def test_shift_and_pred_tile_match(self):
        frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
        shifted = shift_frame(frame, 1, -2)
        tile = pred_tile(frame, 0, 0, 4, 1, -2)
        np.testing.assert_array_equal(tile, shifted[0:4, 0:4])

    def test_motion_stream_roundtrip_with_sparse_and_copy(self):
        classes = np.zeros((3, 8, 8), dtype=np.uint8)
        classes[0, 1:5, 1:5] = 2
        classes[1, 2:6, 2:6] = 2
        classes[2] = classes[1]
        classes[2, 3, 3] = 4

        shifts = make_shifts(search=2, step=1)
        counts, idx = compute_best_motion(classes, block_size=4, shifts=shifts, progress=False)
        records = build_motion_records(
            classes,
            block_size=4,
            shifts=shifts,
            best_counts=counts,
            best_shift_idx=idx,
            sparse_threshold=16,
        )
        streams = pack_motion_streams(classes, records, block_size=4, search=2, step=1)
        decoded = decode_motion_streams(streams)
        np.testing.assert_array_equal(decoded, classes)


if __name__ == "__main__":
    unittest.main()
