"""Deterministic percentile intervals with scenario families as resampling units."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear interpolation between adjacent order statistics."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile requires values and probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_mean_interval(
    clusters: Mapping[str, Sequence[float]], resamples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Resample whole families and recompute a case-weighted mean in each draw.

    Paired comparisons pass within-case score differences as the cluster values,
    preserving both model pairing and dependence among perturbations of a family.
    """
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if not clusters or any(not values for values in clusters.values()):
        raise ValueError("bootstrap requires nonempty clusters")
    summaries = []
    for family in sorted(clusters):
        values = clusters[family]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bootstrap values must be finite")
        summaries.append((sum(values), len(values)))
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draws = [rng.choice(summaries) for _ in summaries]
        means.append(sum(total for total, _ in draws) / sum(n for _, n in draws))
    return percentile(means, 0.025), percentile(means, 0.975)
