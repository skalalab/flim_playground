"""Bounding-box texture extraction matches a full-frame reference.
Padded crops cover textured interior cells, clamp at image borders, and mask out
neighbors in overlapping bounds. Intensity and granularity are bit-identical;
coordinate-derived features allow floating-point rounding.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cell_texture import granularity, radial_distribution, mass_displacement
from src.fov_extraction import extract_texture_features_from_arrays


# Full-frame reference

GRANULARITY_VALUES = [1, 3, 5, 7]
RADIAL_VALUES = [1, 2, 3, 4]


def _full_frame_texture_oracle(intensity_image, mask, fov_name, feature_prefix):
    """Reference implementation: process every cell on the full image."""
    mask_ids = np.unique(mask)
    mask_ids = mask_ids[mask_ids != 0]
    out = {}
    for mask_id in mask_ids:
        cell_id = f"{fov_name}_{mask_id}"
        cell_mask = mask == mask_id
        cell_image = intensity_image * cell_mask  # full-frame, mostly zeros
        feats = {}
        feats[f"{feature_prefix}intensity_sum"] = np.sum(cell_image)
        for n in GRANULARITY_VALUES:
            feats[f"{feature_prefix}granularity_{n}"] = granularity(cell_image, n)
        for ring in RADIAL_VALUES:
            feats[f"{feature_prefix}radial_distribution_ring{ring}"] = radial_distribution(cell_image, ring)
        feats[f"{feature_prefix}mass_displacement"] = mass_displacement(cell_image)
        out[cell_id] = feats
    return pd.DataFrame.from_dict(out, orient="index")


# --- Synthetic frame: interior-textured, edge-touching, adjacent cells -------

def _make_frame():
    """64x64 uint16 frame with four labelled cells exercising the edge cases."""
    H = W = 64
    mask = np.zeros((H, W), dtype=np.uint16)
    intensity = np.zeros((H, W), dtype=np.uint16)
    rng = np.random.default_rng(0)

    def fill(lbl, ys, xs, base):
        mask[ys, xs] = lbl
        # internal texture: base level + a few bright specks so morphological
        # opening actually removes intensity (granularity > 0) and is sensitive
        # to the padded crop margin.
        patch = np.full((ys.stop - ys.start, xs.stop - xs.start), base, np.uint16)
        patch += rng.integers(0, 40, patch.shape).astype(np.uint16)
        patch[1::3, 1::3] += 300  # bright specks
        intensity[ys, xs] = patch

    # 1) interior cell with texture
    fill(1, slice(26, 37), slice(26, 38), base=50)
    # 2) cell flush against the top border (crop must clamp at row 0)
    fill(2, slice(0, 9), slice(6, 16), base=80)
    # 3) & 4) adjacent cells, 1px gap, DIFFERENT intensities (neighbour-bleed)
    fill(3, slice(44, 54), slice(40, 49), base=20)
    fill(4, slice(44, 54), slice(50, 59), base=250)
    return intensity, mask


def test_bbox_crop_matches_full_frame():
    intensity, mask = _make_frame()
    prefix = "Intensity texture_nadh: "

    oracle = _full_frame_texture_oracle(intensity, mask, "FOV1", prefix)
    got = extract_texture_features_from_arrays(intensity, mask, "FOV1", prefix)

    # same cells, same columns, same order
    assert list(got.index) == list(oracle.index)
    assert list(got.columns) == list(oracle.columns)

    # Intensity sum and granularity must be bit-identical.
    exact_cols = [c for c in got.columns if "granularity" in c or "intensity_sum" in c]
    pd.testing.assert_frame_equal(got[exact_cols], oracle[exact_cols], check_exact=True)

    # A crop shifts the coordinate origin, so radial distribution and mass displacement
    # may differ by floating-point rounding (~1e-15).
    fp_cols = [c for c in got.columns if "radial_distribution" in c or "mass_displacement" in c]
    np.testing.assert_allclose(
        got[fp_cols].to_numpy(dtype=float),
        oracle[fp_cols].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12,
    )


def test_adjacent_cell_intensity_does_not_leak():
    """Cell 3's features must be independent of its bright neighbour (cell 4)."""
    intensity, mask = _make_frame()
    prefix = "Intensity texture_nadh: "
    base = extract_texture_features_from_arrays(intensity, mask, "FOV1", prefix)

    # Zero out cell 4 entirely; cell 3's row must be unchanged if no leak occurs.
    intensity2 = intensity.copy()
    intensity2[mask == 4] = 0
    other = extract_texture_features_from_arrays(intensity2, mask, "FOV1", prefix)

    pd.testing.assert_series_equal(base.loc["FOV1_3"], other.loc["FOV1_3"])
