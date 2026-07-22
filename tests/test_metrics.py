"""Unit tests for cv_course.metrics."""

import numpy as np
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from cv_course.metrics import compute_iou, compute_fid, compute_bleu, compute_cer, compute_wer


def test_iou():
    pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    true = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    expected = 3 / 4  # 3 intersect / 4 union
    result = compute_iou(pred, true)
    assert abs(result - expected) < 1e-6, f"IoU: expected {expected}, got {result}"
    print("[OK] test_iou")


def test_iou_empty():
    pred = np.zeros((4, 4), dtype=bool)
    true = np.zeros((4, 4), dtype=bool)
    assert compute_iou(pred, true) == 1.0
    print("[OK] test_iou_empty")


def test_bleu_perfect():
    ref = "the cat sat on the mat".split()
    hyp = "the cat sat on the mat".split()
    bleu = compute_bleu(ref, hyp)
    assert abs(bleu - 1.0) < 1e-4, f"BLEU perfect: expected 1.0, got {bleu}"
    print("[OK] test_bleu_perfect")


def test_bleu_empty():
    assert compute_bleu([], []) == 0.0
    print("[OK] test_bleu_empty")


def test_cer_exact():
    cer = compute_cer("hello", "hello")
    assert cer == 0.0, f"CER exact: expected 0.0, got {cer}"
    print("[OK] test_cer_exact")


def test_cer_substitution():
    cer = compute_cer("kitten", "sitting")
    assert 0.2 < cer < 0.5, f"CER substitution out of range: {cer}"
    print("[OK] test_cer_substitution")


def test_wer_exact():
    wer = compute_wer("the cat sat", "the cat sat")
    assert wer == 0.0, f"WER exact: expected 0.0, got {wer}"
    print("[OK] test_wer_exact")


def test_wer_partial():
    wer = compute_wer("cat sat", "the cat sat on mat")
    assert 0.5 < wer < 1.0, f"WER partial out of range: {wer}"
    print("[OK] test_wer_partial")


if __name__ == "__main__":
    test_iou()
    test_iou_empty()
    test_bleu_perfect()
    test_bleu_empty()
    test_cer_exact()
    test_cer_substitution()
    test_wer_exact()
    test_wer_partial()
    print("\nAll metric tests passed.")
