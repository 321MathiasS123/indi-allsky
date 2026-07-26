import unittest
from unittest import mock

import numpy

from indi_allsky import asi676mc


class TestAsi676mcFrameRepair(unittest.TestCase):

    def test_camera_name_gate(self):
        self.assertTrue(asi676mc.camera_name_matches('ZWO CCD ASI676MC'))
        self.assertTrue(asi676mc.camera_name_matches('ASI-676MC'))
        self.assertTrue(asi676mc.camera_name_matches('ASI676MC 1'))
        self.assertFalse(asi676mc.camera_name_matches('ZWO CCD ASI678MC'))
        self.assertFalse(asi676mc.camera_name_matches(''))

    def test_normal_frame_is_not_modified(self):
        data = numpy.full((64, 64), 1000, dtype=numpy.uint16)
        original = data.copy()

        result = asi676mc.repair_if_needed(data)

        self.assertFalse(result['repaired'])
        numpy.testing.assert_array_equal(data, original)

    def test_bad_frame_is_repaired_in_place(self):
        data = numpy.empty((64, 64), dtype=numpy.uint16)
        data[0::2, 0::2] = 4000
        data[0::2, 1::2] = 1000
        data[1::2, 0::2] = 1000
        data[1::2, 1::2] = 4000

        original_object = data
        result = asi676mc.repair_if_needed(data)

        self.assertTrue(result['repaired'])
        self.assertIs(data, original_object)
        self.assertEqual(data.dtype, numpy.uint16)
        self.assertEqual(data.shape, (64, 64))
        self.assertFalse(result['signature_after']['is_bad'])

    def test_configured_threshold_can_leave_frame_untouched(self):
        data = numpy.empty((64, 64), dtype=numpy.uint16)
        data[0::2, 0::2] = 4000
        data[0::2, 1::2] = 1000
        data[1::2, 0::2] = 1000
        data[1::2, 1::2] = 4000
        original = data.copy()

        result = asi676mc.repair_if_needed(
            data,
            {'PURPLE_RATIO_THRESHOLD': 10.0},
        )

        self.assertFalse(result['repaired'])
        numpy.testing.assert_array_equal(data, original)

    def test_invalid_raw_layout_is_rejected_before_mutation(self):
        odd_width = numpy.zeros((64, 63), dtype=numpy.uint16)
        original = odd_width.copy()

        with self.assertRaises(ValueError):
            asi676mc.repair_if_needed(odd_width)

        numpy.testing.assert_array_equal(odd_width, original)

    def test_failed_validation_retains_original_frame(self):
        data = numpy.empty((64, 64), dtype=numpy.uint16)
        data[0::2, 0::2] = 4000
        data[0::2, 1::2] = 1000
        data[1::2, 0::2] = 1000
        data[1::2, 1::2] = 4000
        original = data.copy()

        with mock.patch.object(asi676mc, 'repair_in_place', side_effect=lambda frame, settings: frame):
            result = asi676mc.repair_if_needed(data)

        self.assertFalse(result['repaired'])
        self.assertTrue(result['validation_failed'])
        numpy.testing.assert_array_equal(data, original)

    def test_chunk_and_sample_sizes_must_preserve_bayer_parity(self):
        with self.assertRaises(ValueError):
            asi676mc.normalize_settings({'SAMPLE_STEP': 3})

        with self.assertRaises(ValueError):
            asi676mc.normalize_settings({'CHUNK_ROWS': 3})


if __name__ == '__main__':
    unittest.main()
