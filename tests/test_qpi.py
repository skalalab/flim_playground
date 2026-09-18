"""src/qpi.py: unit conversion, exclusion geometry, the D14 estimator, diagnostics, per-cell math.

Synthetic 552² backgrounds reproduce the plan's measured facts: the two-sided 2.5·MAD clip is
unbiased (+0.009 nm clean, +0.19 nm with debris) where the one-sided std clip the review called
"reading A" biases by −7 nm. Cell fixtures are small squares so every expected value is hand-checkable.
"""
import numpy as np
import pytest
from scipy.ndimage import binary_dilation
from scipy.stats import entropy, kurtosis, skew
from test_derived_features import CH, _assert_schema_matches

from src import dataset_io, qpi
from src.dataset_io import get_feature_groups_data_extraction
from src.feature_labels import format_feature_label
from src.feature_schema import EXTRACTOR_FEATURE_SUFFIXES
from src.fov_extraction import get_qpi_features


def _synthetic_background(seed=0, shape=(552, 552), sigma_nm=10.0, debris=False):
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    truth_um = 0.030 + 0.020 * xx + 0.015 * yy + 0.010 * xx * yy + 0.008 * xx**2
    image = truth_um + rng.normal(0, sigma_nm * 1e-3, shape)
    if debris:
        hit = rng.random(shape) < 0.04
        image[hit] += rng.exponential(0.060, hit.sum())
    return image, truth_um


def _two_cell_mask(shape=(64, 64)):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[8:24, 8:24] = 1
    mask[36:56, 36:56] = 2
    return mask


def _two_cell_image(mask, cell_um=0.200, noise_um=0.002, seed=0):
    h, w = mask.shape
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    plane = -0.040 + 0.020 * xx - 0.010 * yy
    image = plane + np.where(mask > 0, cell_um, 0.0)
    image += np.random.default_rng(seed).normal(0, noise_um, mask.shape)
    return image, plane


@pytest.mark.parametrize(("unit", "factor"), [("m", 1e6), ("um", 1.0), ("nm", 1e-3)])
def test_to_um_scales_each_unit(unit, factor):
    out = qpi.to_um(np.array([[1.0, -2.0]], dtype=np.float32), unit)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[factor, -2 * factor]])


def test_to_um_rejects_unknown_unit():
    with pytest.raises(ValueError):
        qpi.to_um(np.zeros((2, 2)), "px")


def test_mass_factor_is_pixel_area_over_alpha():
    assert qpi.mass_factor(0.275, 0.181818) == pytest.approx(0.275**2 / 0.181818)
    assert qpi.mass_factor(0.275, 1 / 5.5) == pytest.approx(0.4159375)
    with pytest.raises(ValueError):
        qpi.mass_factor(0.0, 0.181818)


def test_exclusion_region_tight_and_expanded():
    mask = _two_cell_mask()
    assert np.array_equal(qpi.exclusion_region(mask, 0), mask > 0)
    expanded = qpi.exclusion_region(mask, 15)
    assert expanded[mask > 0].all()
    assert expanded.sum() >= 1.15 * (mask > 0).sum()
    grown, steps = mask > 0, 0
    while grown.sum() < expanded.sum():
        grown, steps = binary_dilation(grown), steps + 1
    assert steps >= 1 and np.array_equal(grown, expanded)


def test_clipped_polynomial_is_unbiased_on_clean_gaussian_background():
    image, truth = _synthetic_background()
    surface, info = qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 4)
    assert abs(np.mean(surface - truth)) * 1e3 < 0.5
    assert info["kept_fraction"] > 0.95
    assert 1 <= info["rounds"] <= 10


def test_clipped_polynomial_is_unbiased_with_bright_debris():
    image, truth = _synthetic_background(debris=True)
    surface, info = qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 4)
    assert abs(np.mean(surface - truth)) * 1e3 < 0.5
    assert info["kept_fraction"] > 0.9


def test_one_sided_std_clip_would_fail_the_clean_test():
    image, truth = _synthetic_background()
    h, w = image.shape
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    A = qpi._design_matrix(x.ravel(), y.ravel(), 4)
    z = image.ravel()
    keep = np.ones(z.size, dtype=bool)
    for _ in range(5):
        coeffs = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        resid = z - A @ coeffs
        new_keep = resid < 1.1 * np.std(resid[keep])
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    bias_nm = np.mean((A @ coeffs).reshape(h, w) - truth) * 1e3
    assert bias_nm < -3.0


def test_clipped_polynomial_is_scale_invariant():
    image, _ = _synthetic_background()
    region = np.ones(image.shape, dtype=bool)
    surface, _ = qpi._clipped_polynomial_surface(image, region, 4)
    tiny, _ = qpi._clipped_polynomial_surface(image * 1e-9, region, 4)
    assert np.max(np.abs(tiny / 1e-9 - surface)) / np.max(np.abs(surface)) < 1e-9


def test_clipped_polynomial_clips_both_tails_with_mad():
    rng = np.random.default_rng(1)
    image = 0.050 + rng.normal(0, 0.005, (200, 200))
    hit = rng.random(image.shape)
    image[hit < 0.02] += 0.100
    image[hit > 0.98] -= 0.100
    surface, info = qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 2)
    assert abs(np.mean(surface) - 0.050) * 1e3 < 0.3
    assert 0.94 < info["kept_fraction"] < 0.99


def test_clipped_polynomial_reports_when_the_cap_is_hit():
    image, _ = _synthetic_background(debris=True)
    _, info = qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 4, max_rounds=1)
    assert info["rounds"] == 1
    assert info["kept_fraction"] == 1.0  # the first (and only) fit round used every pixel


def test_clipped_polynomial_rejects_too_small_fit_region():
    image, _ = _synthetic_background(shape=(40, 40))
    region = np.zeros(image.shape, dtype=bool)
    region[:3, :3] = True
    with pytest.raises(ValueError):
        qpi._clipped_polynomial_surface(image, region, 4)


@pytest.mark.parametrize("finite_count", [0, 29])
def test_clipped_polynomial_requires_enough_finite_background_pixels(finite_count):
    image = np.full((20, 20), np.nan)
    image.flat[:finite_count] = np.linspace(-0.04, 0.02, finite_count)
    with pytest.raises(ValueError, match=f"only {finite_count} finite background pixels"):
        qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 1)


def test_clipped_polynomial_kept_fraction_counts_invalid_pixels_as_excluded():
    image, _ = _synthetic_background(shape=(40, 40))
    image[0] = np.nan
    _, info = qpi._clipped_polynomial_surface(image, np.ones(image.shape, dtype=bool), 2, max_rounds=1)
    assert info["kept_fraction"] == pytest.approx(39 / 40)


def test_correct_background_none_returns_zero_surface_and_full_kept():
    mask = _two_cell_mask()
    image, _ = _two_cell_image(mask)
    corrected, surface, exclusion, info = qpi.correct_background(image, mask, {"method": "none", "degree": None, "expand_pct": 0})
    np.testing.assert_array_equal(surface, 0)
    np.testing.assert_array_equal(corrected, image)
    assert np.array_equal(exclusion, mask > 0)
    assert info == {"kept_fraction": 1.0, "rounds": 0}


def test_correct_background_polynomial_removes_a_plane():
    mask = _two_cell_mask()
    image, plane = _two_cell_image(mask)
    corrected, surface, exclusion, info = qpi.correct_background(image, mask, dict(qpi.DEFAULT_BACKGROUND, degree=2))
    background = ~exclusion
    assert abs(np.mean(corrected[background])) * 1e3 < 1.0
    assert np.mean(surface[background] - plane[background]) * 1e3 == pytest.approx(0, abs=1.0)
    assert np.mean(corrected[mask == 1]) == pytest.approx(0.200, abs=0.002)
    assert info["kept_fraction"] > 0.95


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_polynomial_correction_ignores_invalid_background_pixels(invalid_value):
    mask = _two_cell_mask()
    image, plane = _two_cell_image(mask)
    image[0, 0] = invalid_value
    corrected, surface, exclusion, info = qpi.correct_background(image, mask, qpi.DEFAULT_BACKGROUND)
    assert np.isfinite(surface).all()
    assert np.isfinite(corrected[np.isfinite(image)]).all()
    assert not np.isfinite(corrected[0, 0])  # retain the invalid measurement at its original pixel
    np.testing.assert_allclose(surface, plane, atol=0.002)
    assert np.mean(corrected[mask == 1]) == pytest.approx(0.200, abs=0.002)
    diag = qpi.diagnostics(corrected, mask, exclusion, info)
    valid_background = corrected[~exclusion & np.isfinite(corrected)] * 1e3
    np.testing.assert_allclose(diag["bg_percentiles_after_nm"], np.percentile(valid_background, [5, 25, 50, 75, 95]))


def test_correct_background_inpainting_recovers_a_plane_under_cells():
    mask = _two_cell_mask()
    image, _plane = _two_cell_image(mask)
    corrected, _surface, exclusion, info = qpi.correct_background(image, mask, {"method": "inpainting", "degree": None, "expand_pct": 15})
    assert info["kept_fraction"] == 1.0
    assert abs(np.mean(corrected[~exclusion])) * 1e3 < 1.5
    assert np.mean(corrected[mask == 1]) == pytest.approx(0.200, rel=0.05)


def test_correct_background_rejects_unknown_method():
    mask = _two_cell_mask()
    image, _ = _two_cell_image(mask)
    with pytest.raises(ValueError):
        qpi.correct_background(image, mask, {"method": "wavelet", "degree": None, "expand_pct": 0})


def test_diagnostics_reports_background_and_interior_negatives():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[6:14, 6:14] = 1
    corrected = np.zeros((20, 20))
    background = mask == 0
    corrected[background] = np.linspace(-0.010, 0.010, background.sum())
    rim_pixels = [(6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (7, 6), (7, 7), (13, 13), (13, 12), (12, 13), (12, 12)]
    for r, c in rim_pixels:
        corrected[r, c] = -0.001
    for r, c in [(8, 8), (8, 9), (9, 8), (9, 9)]:
        corrected[r, c] = -0.001
    diag = qpi.diagnostics(corrected, mask, mask > 0, {"kept_fraction": 0.9, "rounds": 4})
    assert diag["neg_interior_pct"] == pytest.approx(25.0)
    assert diag["kept_fraction"] == 0.9 and diag["rounds"] == 4
    p5, p25, p50, p75, p95 = diag["bg_percentiles_after_nm"]
    assert p5 < p25 < p50 < p75 < p95
    assert p50 == pytest.approx(0.0, abs=0.05)
    assert p5 == pytest.approx(-9.0, abs=0.2) and p95 == pytest.approx(9.0, abs=0.2)


def test_cell_features_use_signed_sums_and_the_lab_conventions():
    mask = np.zeros((14, 14), dtype=np.uint16)
    mask[2:10, 2:10] = 1                                 # 8x8 = 64 px: large enough for a core and a periphery
    rng = np.random.default_rng(3)
    image = rng.normal(0.2, 0.05, (14, 14))
    image[2, 2] = -0.05                                  # one negative pixel inside the cell
    pixel, alpha = 0.275, 0.181818
    feats = qpi.cell_features(image, mask, pixel, alpha)[1]
    m = image[mask == 1] * pixel**2 / alpha
    assert feats["dry_mass_pg"] == pytest.approx(m.sum())            # signed, not abs
    assert feats["mass_density_pg_per_um2"] == pytest.approx(m.sum() / (64 * pixel**2))
    assert feats["dry_mass_variance"] == pytest.approx(m.var(ddof=1))
    assert feats["dry_mass_skewness"] == pytest.approx(skew(m, bias=False))
    assert feats["dry_mass_kurtosis"] == pytest.approx(kurtosis(m, fisher=True, bias=False))
    pos = np.clip(m, 0, None)
    assert feats["dry_mass_entropy"] == pytest.approx(entropy(pos / pos.sum()))
    assert feats["dry_mass_evenness"] == pytest.approx(entropy(pos / pos.sum()) / np.log(64))
    assert set(feats) == set(qpi.DRY_MASS_SUFFIXES) | set(qpi.SPATIAL_TEXTURE_SUFFIXES)
    assert np.isfinite([feats[s] for s in qpi.SPATIAL_TEXTURE_SUFFIXES]).all()


def test_cell_features_small_samples_are_nan_not_zero():
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[1, 1] = 1
    mask[3, 1:3] = 2
    mask[5, 1:4] = 3
    mask[7, 1:5] = 4
    image = np.random.default_rng(5).normal(0.1, 0.01, (10, 10))   # not constant: moments are defined
    feats = qpi.cell_features(image, mask, 0.275, 0.181818)
    assert np.isnan(feats[1]["dry_mass_variance"]) and np.isfinite(feats[2]["dry_mass_variance"])
    assert np.isnan(feats[2]["dry_mass_skewness"]) and np.isfinite(feats[3]["dry_mass_skewness"])
    assert np.isnan(feats[3]["dry_mass_kurtosis"]) and np.isfinite(feats[4]["dry_mass_kurtosis"])
    assert np.isnan(feats[1]["dry_mass_entropy"]) and np.isnan(feats[1]["dry_mass_evenness"])
    assert np.isfinite(feats[2]["dry_mass_entropy"])
    assert np.isnan(feats[1]["radial_mass_index"])
    assert np.isfinite(feats[1]["dry_mass_pg"])


def test_cell_features_radial_and_polar_indices_guard_nonpositive_totals():
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[3:13, 3:13] = 1
    negative = -0.1 + np.random.default_rng(3).normal(0, 1e-4, (16, 16))
    feats = qpi.cell_features(negative, mask, 0.275, 0.181818)[1]
    assert feats["dry_mass_pg"] < 0
    assert np.isnan(feats["radial_mass_index"])
    assert np.isnan(feats["polar_gradient_index"])
    assert np.isnan(feats["dry_mass_entropy"])
    centred = np.zeros((16, 16))
    centred[6:10, 6:10] = 0.3
    feats = qpi.cell_features(centred, mask, 0.275, 0.181818)[1]
    assert np.isnan(feats["radial_mass_index"])
    ring = np.zeros((16, 16))
    ring[3:13, 3:13] = 0.1
    ring[6:10, 6:10] = 0.3
    feats = qpi.cell_features(ring, mask, 0.275, 0.181818)[1]
    assert feats["radial_mass_index"] > 1
    assert feats["polar_gradient_index"] == pytest.approx(0.0, abs=1e-9)


def test_line_scan_follows_the_major_axis_and_marks_the_mask():
    mask = np.zeros((30, 40), dtype=np.uint16)
    mask[13:17, 5:35] = 1
    raw = np.full((30, 40), -0.040)
    raw[mask == 1] += 0.200
    surface = np.full((30, 40), -0.040)
    corrected = raw - surface
    scan = qpi.line_scan(raw, corrected, surface, mask, 1, 0.275, 0.181818)
    n = scan["position_px"].size
    assert 30 <= n <= 36
    assert scan["inside"].sum() >= 28
    np.testing.assert_allclose(scan["raw_nm"] - scan["surface_nm"], scan["corrected_nm"], atol=1e-6)
    np.testing.assert_allclose(scan["mass_pg"], scan["corrected_nm"] * 1e-3 * qpi.mass_factor(0.275, 0.181818))
    assert scan["corrected_nm"][n // 2] == pytest.approx(200.0)
    with pytest.raises(ValueError):
        qpi.line_scan(raw, corrected, surface, mask, 7, 0.275, 0.181818)


@pytest.mark.parametrize("angle_degrees", [-60, -45, -30, 30, 45, 60])
def test_line_scan_follows_rotated_cells(angle_degrees):
    yy, xx = np.mgrid[:100, :100]
    angle = np.deg2rad(angle_degrees)
    along = (yy - 50) * np.cos(angle) + (xx - 50) * np.sin(angle)
    across = -(yy - 50) * np.sin(angle) + (xx - 50) * np.cos(angle)
    mask = ((along / 30)**2 + (across / 4)**2 <= 1).astype(np.uint16)
    corrected = mask * 0.200
    surface = np.full(mask.shape, -0.040)
    scan = qpi.line_scan(corrected + surface, corrected, surface, mask, 1, 0.275, 0.181818)
    assert 58 <= scan["position_px"].size <= 64
    assert scan["inside"].mean() > 0.9
    # The middle of a lengthwise scan stays within the cell, away from its tips.
    np.testing.assert_allclose(scan["corrected_nm"][5:-5], 200.0, atol=1e-6)


# ---- emitter <-> schema, labels, grouping (spec section 9) ------------------

_CONSTANTS = {"pixel_size_um": 0.275, "opd_unit": "m", "alpha_um3_per_pg": 0.181818}


def _signed_phase():
    """A signed phase array also exercises the negative-value path."""
    return np.random.default_rng(0).normal(0, 0.5, (12, 12))


def _schema_mask():
    mask = np.zeros((12, 12), dtype=int)
    mask[2:6, 2:6] = 1
    mask[7:11, 7:11] = 2
    return mask


def test_schema_suffix_tables_match_the_module():
    assert list(EXTRACTOR_FEATURE_SUFFIXES["Dry-mass statistics"]) == list(qpi.DRY_MASS_SUFFIXES)
    assert list(EXTRACTOR_FEATURE_SUFFIXES["Spatial texture"]) == list(qpi.SPATIAL_TEXTURE_SUFFIXES)


@pytest.mark.parametrize("selected", [
    ["Dry-mass statistics"],
    ["Spatial texture"],
    ["Dry-mass statistics", "Spatial texture"],
    ["Intensity morphology", "Dry-mass statistics", "Spatial texture"],
])
def test_qpi_emitter_matches_schema(selected):
    err, emitted = get_qpi_features(_signed_phase(), _schema_mask(), "fov_1", CH, _CONSTANTS, selected)
    assert err == ""
    qpi_only = [s for s in selected if s != "Intensity morphology"]   # morphology comes from its own extractor
    _assert_schema_matches(emitted, {CH: qpi_only}, {CH: 1})
    assert list(emitted.index) == ["fov_1_1", "fov_1_2"]


def test_qpi_emitter_without_qpi_extractors_is_empty():
    err, emitted = get_qpi_features(_signed_phase(), _schema_mask(), "fov_1", CH, _CONSTANTS, ["Intensity morphology"])
    assert err == "" and emitted.empty


def test_qpi_emitter_reports_shape_mismatch_and_empty_mask():
    err, _ = get_qpi_features(np.zeros((10, 10)), _schema_mask(), "fov_1", CH, _CONSTANTS, ["Dry-mass statistics"])
    assert "different shape" in err
    err, _ = get_qpi_features(_signed_phase(), np.zeros((12, 12), dtype=int), "fov_1", CH, _CONSTANTS, ["Dry-mass statistics"])
    assert "No cells" in err


@pytest.mark.parametrize(("column", "expected"), [
    ("Dry-mass statistics_QPI: dry_mass_pg", "QPI Dry mass (pg)"),
    ("Dry-mass statistics_QPI: mass_density_pg_per_um2", "QPI Mass density (pg/µm²)"),
    ("Dry-mass statistics_QPI: dry_mass_variance", "QPI Mass variance (pg²)"),
    ("Dry-mass statistics_QPI: dry_mass_skewness", "QPI Mass skewness"),
    ("Dry-mass statistics_QPI: dry_mass_kurtosis", "QPI Mass kurtosis"),
    ("Dry-mass statistics_QPI: dry_mass_entropy", "QPI Mass entropy"),
    ("Dry-mass statistics_QPI: dry_mass_evenness", "QPI Mass evenness"),
    ("Spatial texture_QPI: avg_dm_gradient_mag", "QPI Mass gradient (pg/px)"),
    ("Spatial texture_QPI: radial_mass_index", "QPI Radial mass index"),
    ("Spatial texture_QPI: polar_gradient_index", "QPI Polar gradient index"),
])
def test_qpi_feature_labels(column, expected):
    assert format_feature_label(column) == expected


def test_qpi_columns_group_by_extractor_and_channel(monkeypatch):
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors", lambda: [
        "Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture",
        "Dry-mass statistics", "Spatial texture"])
    groups = get_feature_groups_data_extraction([
        "Dry-mass statistics_QPI: dry_mass_pg", "Spatial texture_QPI: radial_mass_index",
        "Intensity morphology_QPI: area", "QPI_centroid_x"])
    assert groups["Dry-mass statistics_QPI"] == ["Dry-mass statistics_QPI: dry_mass_pg"]
    assert groups["Spatial texture_QPI"] == ["Spatial texture_QPI: radial_mass_index"]
    assert groups["Intensity morphology_QPI"] == ["Intensity morphology_QPI: area"]
    assert groups["Uncategorized Features"] == ["QPI_centroid_x"]


def test_qpi_columns_land_in_uncategorized_without_the_migrated_list(monkeypatch):
    """Why main.py migrates saved all_feature_extractors lists (spec section 4, Task 3)."""
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"])
    groups = get_feature_groups_data_extraction(["Dry-mass statistics_QPI: dry_mass_pg"])
    assert groups == {"Uncategorized Features": ["Dry-mass statistics_QPI: dry_mass_pg"]}
