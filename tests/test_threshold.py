# tests/test_threshold.py
"""
Tests for threshold selection logic.

The test suite focuses on verifying that the threshold selection procedure
behaves reasonably on a simple synthetic example, where score ordering and
false positive rate can be controlled explicitly.
"""

import numpy as np

from src.models.train import find_threshold_by_target_fpr


def test_find_threshold_by_target_fpr_simple_case():
    """
    The test checks threshold selection against a simple synthetic scenario.

    The input scores are ordered such that increasing the threshold gradually
    reduces the false positive rate, allowing verification that the selected
    threshold produces an FPR close to the specified target.
    """

    # Ground-truth labels where 0 denotes negative class and 1 denotes positive class
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Model scores where higher values indicate higher confidence in the positive class
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    # Target false positive rate used for threshold selection
    target_fpr = 0.25

    threshold, info = find_threshold_by_target_fpr(
        y_true = y_true,
        y_score = y_score,
        target_fpr = target_fpr,
    )

    # The selected threshold is expected to lie within the score range
    assert 0.0 <= threshold <= 1.0

    # The returned metadata is expected to include the achieved false positive rate
    assert "FPR" in info

    # The achieved FPR is expected to be reasonably close to the target value
    assert abs(info["FPR"] - target_fpr) < 0.15
