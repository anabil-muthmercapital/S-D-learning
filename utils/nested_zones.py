# =============================================================================
# utils/nested_zones.py — Nested zone detection & merging (Phase 13)
# =============================================================================
#
# Responsibilities
# ----------------
# find_nested_pairs() — identify pairs of same-type zones that overlap
#                       significantly on the price axis.
# merge_zones()       — collapse each overlapping pair into one merged zone.
#
# Overlap ratio
# -------------
#   overlap       = max(0, min(prox1,prox2) − max(dist1,dist2))   [demand]
#   overlap_ratio = overlap / min(width1, width2)
#
# If overlap_ratio >= NESTED_OVERLAP_MIN → the pair is nested and merged.
#
# Merge rule (same type only)
# ---------------------------
#   Demand — merged_proximal = min(prox1, prox2)  ← sharper (lower) entry
#             merged_distal   = min(dist1, dist2)  ← safer  (lower) stop
#   Supply — merged_proximal = max(prox1, prox2)  ← sharper (higher) entry
#             merged_distal   = max(dist1, dist2)  ← safer  (higher) stop
#
# The merged dict inherits all fields from the *first* zone of the pair and
# overwrites proximal/distal/zone_width.  A ``"nested": True`` flag is added.
#
# Prerequisites
# -------------
# zones must have proximal, distal, zone_width, zone_type (from detect_zones).
# All other scoring annotations are preserved on the merged zone.
# =============================================================================

from __future__ import annotations

from utils.config import NESTED_OVERLAP_MIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlap_ratio(z1: dict, z2: dict) -> float:
    """Overlap between the price intervals of two zones, normalised by the
    width of the smaller zone.

    Returns a value in [0, 1] (may exceed 1 only if both have identical width
    and are fully overlapping, but practically stays ≤ 1).
    """
    lo1 = min(z1["proximal"], z1["distal"])
    hi1 = max(z1["proximal"], z1["distal"])
    lo2 = min(z2["proximal"], z2["distal"])
    hi2 = max(z2["proximal"], z2["distal"])

    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    min_width = min(hi1 - lo1, hi2 - lo2)
    return overlap / min_width if min_width > 0 else 0.0


def _merge_pair(z1: dict, z2: dict) -> dict:
    """Return a new zone dict that is the merge of *z1* and *z2*.

    All fields are copied from *z1*; only proximal, distal, and zone_width
    are recomputed.  A ``nested`` flag is set to True.
    """
    if z1["zone_type"] == "demand":
        merged_prox = min(z1["proximal"], z2["proximal"])  # sharper (lower) entry
        merged_dist = min(z1["distal"], z2["distal"])  # safer  (lower) stop
    else:
        merged_prox = max(z1["proximal"], z2["proximal"])  # sharper (higher) entry
        merged_dist = max(z1["distal"], z2["distal"])  # safer  (higher) stop

    return {
        **z1,
        "proximal": merged_prox,
        "distal": merged_dist,
        "zone_width": round(abs(merged_prox - merged_dist), 5),
        "nested": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_nested_pairs(zones: list[dict]) -> list[tuple[int, int]]:
    """Return index pairs (i, j) where *i < j* and the two zones overlap.

    Only same-type (demand–demand or supply–supply) pairs are considered.
    The overlap ratio threshold is taken from ``NESTED_OVERLAP_MIN`` in config.

    Parameters
    ----------
    zones : list of zone dicts, each with proximal, distal, zone_type.

    Returns
    -------
    List of (i, j) tuples sorted by i then j.
    """
    pairs: list[tuple[int, int]] = []
    for i, z1 in enumerate(zones):
        for j, z2 in enumerate(zones):
            if j <= i:
                continue
            if z1["zone_type"] != z2["zone_type"]:
                continue
            if _overlap_ratio(z1, z2) >= NESTED_OVERLAP_MIN:
                pairs.append((i, j))
    return pairs


def merge_zones(zones: list[dict]) -> list[dict]:
    """Merge all nested pairs and return the de-duplicated zone list.

    For every nested pair (i, j):
      - zones[i] and zones[j] are removed from the output
      - a new merged zone (inheriting from zones[i]) is appended

    If a zone participates in more than one pair it is still only removed
    once; the first pair it appears in drives the merge.

    Parameters
    ----------
    zones : list of zone dicts (already scored or not — doesn't matter).

    Returns
    -------
    New list with overlapping pairs replaced by merged zones.
    ``z["nested"] == True`` on every merged entry.
    ``z["nested"]`` is absent (or False) on untouched zones.
    """
    pairs = find_nested_pairs(zones)

    merged_idxs: set[int] = set()
    merged_zones: list[dict] = []

    for i, j in pairs:
        # Each original index is only merged once (first pair wins)
        if i in merged_idxs or j in merged_idxs:
            continue
        merged_zones.append(_merge_pair(zones[i], zones[j]))
        merged_idxs.update([i, j])

    result = [z for k, z in enumerate(zones) if k not in merged_idxs]
    result.extend(merged_zones)
    return result
