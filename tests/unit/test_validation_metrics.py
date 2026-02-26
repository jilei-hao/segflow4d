"""Unit tests for segmentation validation metrics (Dice, MSD, HD95)."""

import numpy as np
import pytest

from segflow4d.utility.validation.segmentation_validation import (
    evaluate_segmentation,
    ValidationResult,
    Metrics,
)

SPACING = (1.0, 1.0, 1.0)


def _make_box_seg(shape, label, start, end):
    """Return an integer array with a filled box labelled `label`."""
    arr = np.zeros(shape, dtype=np.int32)
    arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = label
    return arr


class TestDiceScore:
    def test_dice_identical_segmentations(self):
        arr = _make_box_seg((20, 20, 20), 1, (5, 5, 5), (15, 15, 15))
        result = evaluate_segmentation(arr, arr, spacing=SPACING)
        assert result.per_label[1].dice == pytest.approx(1.0)

    def test_dice_no_overlap(self):
        ref = _make_box_seg((20, 20, 20), 1, (0, 0, 0), (5, 5, 5))
        target = _make_box_seg((20, 20, 20), 1, (15, 15, 15), (20, 20, 20))
        result = evaluate_segmentation(target, ref, spacing=SPACING)
        assert result.per_label[1].dice == pytest.approx(0.0)

    def test_dice_partial_overlap_known_value(self):
        """
        ref:    voxels [0:10] in z   → 10×10×10 = 1000
        target: voxels [5:15] in z   → 10×10×10 = 1000
        overlap:                          5×10×10 = 500
        Dice = 2*500 / (1000+1000) = 0.5
        """
        shape = (20, 10, 10)
        ref = _make_box_seg(shape, 1, (0, 0, 0), (10, 10, 10))
        target = _make_box_seg(shape, 1, (5, 0, 0), (15, 10, 10))
        result = evaluate_segmentation(target, ref, spacing=SPACING)
        assert result.per_label[1].dice == pytest.approx(0.5, abs=1e-3)

    def test_dice_multi_label_macro_avg(self):
        shape = (20, 20, 20)
        # label 1: perfect match, label 2: no overlap
        ref1 = _make_box_seg(shape, 1, (0, 0, 0), (5, 5, 5))
        ref2 = _make_box_seg(shape, 2, (15, 15, 15), (20, 20, 20))
        ref = ref1 + ref2

        tgt1 = _make_box_seg(shape, 1, (0, 0, 0), (5, 5, 5))   # identical
        tgt2 = _make_box_seg(shape, 2, (0, 15, 15), (5, 20, 20))  # no overlap
        target = tgt1 + tgt2

        result = evaluate_segmentation(target, ref, spacing=SPACING)
        expected_macro = (result.per_label[1].dice + result.per_label[2].dice) / 2
        assert result.macro_avg.dice == pytest.approx(expected_macro, abs=1e-6)

    def test_both_empty_gives_perfect_dice(self):
        """Both empty → conventionally Dice = 1.0."""
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        result = evaluate_segmentation(arr, arr, spacing=SPACING, labels=[1])
        assert result.per_label[1].dice == pytest.approx(1.0)


class TestHD95:
    def test_hd95_zero_for_identical(self):
        arr = _make_box_seg((20, 20, 20), 1, (5, 5, 5), (15, 15, 15))
        result = evaluate_segmentation(arr, arr, spacing=SPACING)
        assert result.per_label[1].hd95 == pytest.approx(0.0)

    def test_hd95_positive_for_partial_overlap(self):
        ref = _make_box_seg((30, 10, 10), 1, (0, 0, 0), (10, 10, 10))
        target = _make_box_seg((30, 10, 10), 1, (5, 0, 0), (15, 10, 10))
        result = evaluate_segmentation(target, ref, spacing=SPACING)
        assert result.per_label[1].hd95 > 0.0


class TestValidationStructure:
    def test_result_has_required_fields(self):
        arr = _make_box_seg((10, 10, 10), 1, (2, 2, 2), (8, 8, 8))
        result = evaluate_segmentation(arr, arr, spacing=SPACING)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.macro_avg, Metrics)
        assert 1 in result.per_label
        assert 1 in result.ref_voxels
        assert 1 in result.target_voxels

    def test_shape_mismatch_raises(self):
        a = np.zeros((10, 10, 10), dtype=np.int32)
        b = np.zeros((10, 10, 9), dtype=np.int32)
        with pytest.raises(ValueError, match="shape"):
            evaluate_segmentation(a, b, spacing=SPACING)
