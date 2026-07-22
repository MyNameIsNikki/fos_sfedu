import numpy as np
from typing import Sequence


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0
    return intersection / union


def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1, mu2 = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean = sigma1 @ sigma2
    eigvals = np.linalg.eigvalsh(covmean)
    covmean_sqrt = np.sqrt(np.maximum(eigvals, 0))
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * np.diag(covmean_sqrt))
    return float(fid)


def compute_bleu(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    max_n: int = 4,
) -> float:
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len == 0 or ref_len == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = set(
            tuple(reference[i : i + n]) for i in range(max(0, ref_len - n + 1))
        )
        hyp_ngrams = [
            tuple(hypothesis[i : i + n]) for i in range(max(0, hyp_len - n + 1))
        ]
        count = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        total = max(len(hyp_ngrams), 1)
        precisions.append(count / total)

    bleu = np.exp(np.mean(np.log(np.maximum(precisions, 1e-10))))
    bp = min(1.0, np.exp(1 - ref_len / max(hyp_len, 1)))
    return float(bleu * bp)


def compute_cer(hyp: str, ref: str) -> float:
    from Levenshtein import distance as lev_dist

    d = lev_dist(hyp, ref)
    return d / max(len(ref), 1)


def compute_wer(hyp: str, ref: str) -> float:
    h_words = hyp.split()
    r_words = ref.split()

    d = np.zeros((len(h_words) + 1, len(r_words) + 1), dtype=np.int32)
    for i in range(len(h_words) + 1):
        d[i, 0] = i
    for j in range(len(r_words) + 1):
        d[0, j] = j

    for i in range(1, len(h_words) + 1):
        for j in range(1, len(r_words) + 1):
            cost = 0 if h_words[i - 1] == r_words[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + cost,
            )
    return d[len(h_words), len(r_words)] / max(len(r_words), 1)
