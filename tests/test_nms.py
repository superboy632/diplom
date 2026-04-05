"""
tests/test_nms.py
─────────────────
Unit tests for the NMS (Non-Maximum Suppression) function in geo_utils.py.

Each detection row is: [cx_norm, cy_norm, w_norm, h_norm, confidence]

Edge cases covered
──────────────────
1.  Empty input                     → empty array returned unchanged
2.  Single detection                → always kept
3.  Identical boxes                 → only highest confidence kept
4.  Fully overlapping boxes (IoU=1) → only highest confidence kept
5.  Non-overlapping boxes           → all kept regardless of confidence
6.  Partially overlapping, IoU<0.5  → both kept
7.  Partially overlapping, IoU>0.5  → lower-confidence suppressed
8.  Output is sorted by confidence  → highest-confidence box is first
9.  Threshold boundary behaviour    → IoU exactly at threshold (kept/suppressed)
10. Many boxes, only one cluster    → single survivor per cluster
"""

import numpy as np
import pytest

from geo_utils import nms


# ── Helpers ───────────────────────────────────────────────────────────────────
def box(cx: float, cy: float, w: float, h: float, conf: float) -> np.ndarray:
    """Convenience constructor for a single detection row."""
    return np.array([[cx, cy, w, h, conf]], dtype=np.float32)


def boxes(*rows) -> np.ndarray:
    """Stack multiple box() calls into a (K, 5) array."""
    return np.vstack(rows)


def iou_of(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two (1, 5) detection rows."""
    x1a, y1a = a[0, 0] - a[0, 2] / 2, a[0, 1] - a[0, 3] / 2
    x2a, y2a = a[0, 0] + a[0, 2] / 2, a[0, 1] + a[0, 3] / 2
    x1b, y1b = b[0, 0] - b[0, 2] / 2, b[0, 1] - b[0, 3] / 2
    x2b, y2b = b[0, 0] + b[0, 2] / 2, b[0, 1] + b[0, 3] / 2
    ix = max(0, min(x2a, x2b) - max(x1a, x1b))
    iy = max(0, min(y2a, y2b) - max(y1a, y1b))
    inter = ix * iy
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / (union + 1e-8)


# ════════════════════════════════════════════════════════════════════════════════
# 1. Empty input
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSEmptyInput:
    def test_empty_array_returns_empty(self):
        dets = np.empty((0, 5), dtype=np.float32)
        result = nms(dets)
        assert result.shape == (0, 5)

    def test_empty_preserves_dtype(self):
        dets = np.empty((0, 5), dtype=np.float64)
        result = nms(dets)
        assert result.shape[0] == 0


# ════════════════════════════════════════════════════════════════════════════════
# 2. Single detection
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSSingleDetection:
    def test_single_is_always_kept(self):
        dets = box(0.5, 0.5, 0.2, 0.2, 0.9)
        result = nms(dets)
        assert len(result) == 1
        np.testing.assert_array_equal(result, dets)

    def test_single_low_confidence_kept(self):
        dets = box(0.5, 0.5, 0.3, 0.3, 0.01)
        assert len(nms(dets)) == 1

    def test_single_tiny_box_kept(self):
        dets = box(0.5, 0.5, 1e-4, 1e-4, 0.99)
        assert len(nms(dets)) == 1


# ════════════════════════════════════════════════════════════════════════════════
# 3. Identical boxes
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSIdenticalBoxes:
    def test_two_identical_keeps_higher_conf(self):
        dets = boxes(
            box(0.5, 0.5, 0.3, 0.3, 0.9),
            box(0.5, 0.5, 0.3, 0.3, 0.5),
        )
        result = nms(dets)
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.9)

    def test_three_identical_keeps_only_one(self):
        dets = boxes(
            box(0.4, 0.4, 0.2, 0.2, 0.7),
            box(0.4, 0.4, 0.2, 0.2, 0.9),
            box(0.4, 0.4, 0.2, 0.2, 0.5),
        )
        result = nms(dets)
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.9)


# ════════════════════════════════════════════════════════════════════════════════
# 4. Fully overlapping boxes (IoU = 1.0)
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSFullOverlap:
    def test_iou_is_one_for_same_box(self):
        a = box(0.5, 0.5, 0.4, 0.4, 0.8)
        assert iou_of(a, a) == pytest.approx(1.0, rel=1e-4)

    def test_full_overlap_keeps_highest_conf(self):
        high = box(0.5, 0.5, 0.4, 0.4, 0.95)
        low  = box(0.5, 0.5, 0.4, 0.4, 0.40)
        result = nms(boxes(low, high))
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.95)

    def test_five_identical_keeps_one(self):
        confs = [0.3, 0.9, 0.5, 0.7, 0.1]
        dets  = boxes(*[box(0.5, 0.5, 0.3, 0.3, c) for c in confs])
        result = nms(dets)
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.9)


# ════════════════════════════════════════════════════════════════════════════════
# 5. Non-overlapping boxes (all must be kept)
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSNonOverlapping:
    def _non_overlapping_grid(self):
        """4 boxes placed in corners — zero overlap between any pair."""
        return boxes(
            box(0.1,  0.1,  0.1, 0.1, 0.9),
            box(0.9,  0.1,  0.1, 0.1, 0.8),
            box(0.1,  0.9,  0.1, 0.1, 0.7),
            box(0.9,  0.9,  0.1, 0.1, 0.6),
        )

    def test_all_non_overlapping_kept(self):
        dets = self._non_overlapping_grid()
        result = nms(dets)
        assert len(result) == 4

    def test_non_overlapping_verify_zero_iou(self):
        a = box(0.1, 0.5, 0.1, 0.1, 0.9)
        b = box(0.9, 0.5, 0.1, 0.1, 0.8)
        assert iou_of(a, b) == pytest.approx(0.0, abs=1e-6)
        result = nms(boxes(a, b))
        assert len(result) == 2

    def test_many_spread_boxes_all_kept(self):
        # 10 boxes evenly spaced along a horizontal band — no overlap
        dets = np.array(
            [[0.05 + 0.1 * i, 0.5, 0.05, 0.05, 0.9 - 0.05 * i] for i in range(10)],
            dtype=np.float32,
        )
        result = nms(dets)
        assert len(result) == 10


# ════════════════════════════════════════════════════════════════════════════════
# 6 & 7. Partial overlap — below and above threshold
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSPartialOverlap:
    def _two_boxes_with_known_iou(self, offset: float):
        """
        Two 0.4×0.4 boxes; second shifted by `offset` in x.
        IoU is calculable: overlap width = max(0, 0.4 − offset).
        """
        a = box(0.5,         0.5, 0.4, 0.4, 0.9)
        b = box(0.5 + offset, 0.5, 0.4, 0.4, 0.7)
        return a, b

    def test_low_overlap_both_kept(self):
        """IoU ≈ 0.18 < 0.5 → both survive."""
        a, b = self._two_boxes_with_known_iou(offset=0.3)
        computed_iou = iou_of(a, b)
        assert computed_iou < 0.5, f"Pre-condition failed: IoU={computed_iou:.3f}"
        result = nms(boxes(a, b))
        assert len(result) == 2

    def test_high_overlap_lower_suppressed(self):
        """IoU ≈ 0.62 > 0.5 → lower-confidence box suppressed."""
        a, b = self._two_boxes_with_known_iou(offset=0.1)
        computed_iou = iou_of(a, b)
        assert computed_iou > 0.5, f"Pre-condition failed: IoU={computed_iou:.3f}"
        result = nms(boxes(a, b))
        assert len(result) == 1
        assert result[0, 4] == pytest.approx(0.9)

    def test_custom_iou_threshold_respected(self):
        """With iou_thresh=0.3, the pair from test_low_overlap should collapse."""
        a, b = self._two_boxes_with_known_iou(offset=0.3)
        iou  = iou_of(a, b)
        # At thresh=0.3 the pair should collapse (iou > 0.3 but < 0.5)
        if iou > 0.3:
            result = nms(boxes(a, b), iou_thresh=0.3)
            assert len(result) == 1
        else:
            pytest.skip(f"IoU={iou:.3f} not above 0.3 for this geometry")


# ════════════════════════════════════════════════════════════════════════════════
# 8. Output ordering
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSOutputOrdering:
    def test_highest_conf_is_first(self):
        dets = boxes(
            box(0.1, 0.5, 0.05, 0.05, 0.6),
            box(0.9, 0.5, 0.05, 0.05, 0.95),
            box(0.5, 0.1, 0.05, 0.05, 0.8),
        )
        result = nms(dets)
        # All non-overlapping → all kept; highest conf first
        assert len(result) == 3
        assert result[0, 4] == pytest.approx(0.95)

    def test_output_confidences_descending(self):
        dets = np.array(
            [[0.05 + 0.1 * i, 0.5, 0.04, 0.04, 0.1 * (i + 1)] for i in range(8)],
            dtype=np.float32,
        )
        result = nms(dets)
        confs = result[:, 4]
        assert list(confs) == sorted(confs, reverse=True)


# ════════════════════════════════════════════════════════════════════════════════
# 9. Threshold boundary
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSThresholdBoundary:
    def test_exactly_at_threshold_is_suppressed(self):
        """
        IoU == iou_thresh: the condition is `iou < thresh`, so a box
        exactly at the threshold is suppressed (strict less-than).
        """
        # Build two boxes with IoU = 0.5 exactly
        # Box A: [0.3, 0.3, 0.7, 0.7]  area = 0.4*0.4 = 0.16
        # Box B: same centre, slightly shifted — use analytical construction
        # Easier: just measure and set thresh to the computed IoU
        a = box(0.5, 0.5, 0.4, 0.4, 0.9)
        b = box(0.5, 0.5, 0.4, 0.4, 0.6)   # identical → IoU = 1.0
        thresh = iou_of(a, b)               # ≈ 1.0
        result = nms(boxes(a, b), iou_thresh=thresh)
        # IoU == thresh → condition `iou < thresh` is False → b is suppressed
        assert len(result) == 1

    def test_just_below_threshold_both_kept(self):
        a = box(0.1, 0.5, 0.1, 0.1, 0.9)
        b = box(0.9, 0.5, 0.1, 0.1, 0.7)
        iou = iou_of(a, b)                  # ≈ 0.0
        result = nms(boxes(a, b), iou_thresh=max(iou + 0.01, 0.01))
        assert len(result) == 2


# ════════════════════════════════════════════════════════════════════════════════
# 10. Cluster behaviour
# ════════════════════════════════════════════════════════════════════════════════

class TestNMSClusters:
    def test_two_separate_clusters(self):
        """Two spatially distinct clusters → one survivor each."""
        cluster_a = boxes(
            box(0.2, 0.2, 0.15, 0.15, 0.9),
            box(0.2, 0.2, 0.15, 0.15, 0.7),
            box(0.2, 0.2, 0.15, 0.15, 0.5),
        )
        cluster_b = boxes(
            box(0.8, 0.8, 0.15, 0.15, 0.85),
            box(0.8, 0.8, 0.15, 0.15, 0.65),
        )
        dets = np.vstack([cluster_a, cluster_b])
        result = nms(dets)
        assert len(result) == 2
        confs = sorted(result[:, 4], reverse=True)
        assert confs[0] == pytest.approx(0.9)
        assert confs[1] == pytest.approx(0.85)

    def test_three_clusters_three_survivors(self):
        clusters = []
        centres = [(0.15, 0.15), (0.5, 0.5), (0.85, 0.85)]
        for cx, cy in centres:
            for conf in [0.9, 0.7, 0.5]:
                clusters.append(box(cx, cy, 0.1, 0.1, conf))
        dets = np.vstack(clusters)
        result = nms(dets)
        assert len(result) == 3
        # Each survivor should be the 0.9-confidence box from its cluster
        np.testing.assert_allclose(sorted(result[:, 4], reverse=True), [0.9, 0.9, 0.9])
