# =============================================================================
# utils/nested_zones.py — Nested zone detection & merging (Phase 13)
# =============================================================================
#
# Responsibilities
# ----------------
# find_nested_pairs() — identify pairs of same-type zones that overlap
#                       significantly on the price axis (diagnostics).
# merge_zones()       — collapse each connected component of transitively
#                       overlapping same-type zones into one merged zone.
#
# Overlap ratio
# -------------
#   overlap       = max(0, min(prox1,prox2) − max(dist1,dist2))   [demand]
#   overlap_ratio = overlap / min(width1, width2)
#
# If overlap_ratio >= NESTED_OVERLAP_MIN → the pair is nested.
#
# Merge rule (same type only, generalised to N zones per component)
# -----------------------------------------------------------------
#   Demand — merged_proximal = min(proximals)   ← sharper (lower) entry
#             merged_distal   = min(distals)     ← safer  (lower) stop
#   Supply — merged_proximal = max(proximals)   ← sharper (higher) entry
#             merged_distal   = max(distals)     ← safer  (higher) stop
#
# The merged dict inherits all non-geometry fields from the FIRST zone in the
# component (lowest original index) and overwrites proximal/distal/zone_width.
# A ``"nested": True`` flag is added on every merged entry.
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


def _merge_many(group: list[dict]) -> dict:
    """Return a single zone dict that is the merge of every zone in *group*.

    Generalises ``_merge_pair`` to N same-type zones. All non-geometry fields
    are inherited from the FIRST zone (the lowest original index in the
    connected component, preserved by the caller).
    """
    if len(group) == 1:
        return group[0]

    base = group[0]
    proximals = [z["proximal"] for z in group]
    distals = [z["distal"] for z in group]

    if base["zone_type"] == "demand":
        merged_prox = min(proximals)  # sharper (lower) entry
        merged_dist = min(distals)  # safer  (lower) stop
    else:
        merged_prox = max(proximals)  # sharper (higher) entry
        merged_dist = max(distals)  # safer  (higher) stop

    return {
        **base,
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
    """Merge transitively-overlapping zones via connected components.

    Why connected components
    ------------------------
    The previous "first pair wins" rule under-merged: a zone A overlapping
    both B and C would only merge with B, leaving C as a separate (still
    overlapping) zone. Building a graph of overlap edges and merging each
    connected component into one zone guarantees that any cluster of mutually
    or transitively overlapping same-type zones collapses to a single zone.

    Procedure
    ---------
      1. Edges: same-type pairs (i, j) with overlap ≥ NESTED_OVERLAP_MIN.
      2. Find connected components via union-find.
      3. Each component of size > 1 → one merged zone (``_merge_many``,
         ``nested = True``). Components of size 1 pass through unchanged.

    Output order follows the lowest original index of each component, so the
    result is stable and roughly preserves input order.
    """
    n = len(zones)
    if n == 0:
        return []

    # Union-find over zone indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the smaller index as the root so component leader == first zone.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, j in find_nested_pairs(zones):
        union(i, j)

    # Group indices by their root, preserving original order within each group.
    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    # Emit one zone per component, ordered by the component leader (lowest idx).
    result: list[dict] = []
    for root in sorted(components):
        group = [zones[k] for k in components[root]]
        result.append(_merge_many(group))
    return result
