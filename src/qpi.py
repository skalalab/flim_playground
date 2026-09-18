"""Quantitative phase imaging (QPI): background correction, diagnostics and per-cell dry-mass math.

Streamlit-free and free of project imports (numpy / scipy / scikit-image only), mirroring
``src/cell_texture.py``, so every function is unit-testable directly. A port of the lab's
cleaned ``qpi_calibration.py`` and ``qpi_single_cell_measurements.py`` (Tentellino, Desa,
Sheinmann): ``exclusion_region`` (their ``expand_mask_by_area``) and the inpainting method
follow the lab. Differences are marked ``DIFFERS FROM LAB`` where
they occur: the polynomial estimator (design D14, a
two-sided 2.5 x MAD clip iterated to convergence instead of plain least squares on every
tenth pixel), the unit convention (OPD in micrometres; mass per pixel = OPD_um x pixel
area / alpha, in pg) and signed sums with NaN for undefined small-sample statistics
(design D13, spec section 8) instead of ``np.abs`` and 0 / 1 fallbacks, and the corrected
major-axis direction for line scans in image coordinates.
"""
import math

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    find_objects,
    gaussian_filter,
)
from scipy.stats import entropy, kurtosis, skew
from skimage.filters import scharr
from skimage.measure import profile_line, regionprops
from skimage.restoration import inpaint_biharmonic

OPD_UNITS = ("m", "um", "nm")
OPD_TO_UM = {"m": 1_000_000.0, "um": 1.0, "nm": 0.001}
BACKGROUND_METHODS = ("none", "polynomial", "inpainting")
DEFAULT_BACKGROUND = {"method": "polynomial", "degree": 4, "expand_pct": 15.0}
# Below this share of kept background pixels the clipped fit may sit too low (masses inflated).
KEPT_FRACTION_WARNING = 0.85

# Feature suffixes in emission order. src/feature_schema.py lists the same names; tests/test_qpi.py checks.
DRY_MASS_SUFFIXES = (
    "dry_mass_pg", "mass_density_pg_per_um2", "dry_mass_variance", "dry_mass_skewness",
    "dry_mass_kurtosis", "dry_mass_entropy", "dry_mass_evenness",
)
SPATIAL_TEXTURE_SUFFIXES = ("avg_dm_gradient_mag", "radial_mass_index", "polar_gradient_index")


def to_um(image, opd_unit):
    """Convert a stored OPD image to float64 micrometres (DIFFERS FROM LAB: metres throughout)."""
    if opd_unit not in OPD_TO_UM:
        raise ValueError(f"Unknown OPD unit {opd_unit!r}; expected one of {', '.join(OPD_UNITS)}")
    return np.asarray(image, dtype=np.float64) * OPD_TO_UM[opd_unit]


def mass_factor(pixel_size_um, alpha_um3_per_pg):
    """Picograms of dry mass per micrometre of OPD in one pixel: pixel area / alpha.

    The lab's ``dry_mass_conversion_factor`` is this times 1e6 because its OPD is in metres.
    """
    if not (math.isfinite(pixel_size_um) and pixel_size_um > 0):
        raise ValueError("pixel_size_um must be finite and greater than zero")
    if not (math.isfinite(alpha_um3_per_pg) and alpha_um3_per_pg > 0):
        raise ValueError("alpha_um3_per_pg must be finite and greater than zero")
    return float(pixel_size_um) ** 2 / float(alpha_um3_per_pg)


def exclusion_region(mask, expand_pct):
    """Iteratively dilate the cell mask until its area grows by expand_pct (lab's expand_mask_by_area)."""
    base_mask = (mask > 0)
    if expand_pct <= 0:
        return base_mask
    original_area = np.sum(base_mask)
    target_area = original_area * (1.0 + (expand_pct / 100.0))
    expanded_mask = base_mask.copy()
    while np.sum(expanded_mask) < target_area:
        expanded_mask = binary_dilation(expanded_mask)
        if np.sum(expanded_mask) == expanded_mask.size:
            break
    return expanded_mask


def _design_matrix(x, y, degree):
    """Monomials x**i * y**j with i + j <= degree (the lab's get_poly_features)."""
    return np.column_stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)])


def _clipped_polynomial_surface(image, fit_region, degree, k=2.5, max_rounds=10):
    """DIFFERS FROM LAB (design D14): iterative two-sided robust-clip least squares.

    Residuals are taken over the finite fit region every round; the scale is 1.4826 x MAD
    around their median, never a standard deviation and never over the kept subset (a
    one-sided std clip biases a clean background by -4 to -10 nm). Pixels beyond k x scale
    on either side are dropped and the fit repeated until the kept set is unchanged or
    max_rounds is reached, in which case the last fit is accepted. Returns
    ``(surface, {"kept_fraction", "rounds"})``; the kept fraction is the share of the fit
    region actually used by the fit that produced the returned surface, counting invalid
    pixels as excluded (never a later, unfitted round's set).
    """
    h, w = image.shape
    background_count = int(fit_region.sum())
    fit_region = fit_region & np.isfinite(image)
    finite_count = int(fit_region.sum())
    n_coefficients = (degree + 1) * (degree + 2) // 2
    if finite_count < 10 * n_coefficients:
        raise ValueError(
            f"only {finite_count} finite background pixels for a degree-{degree} surface; "
            "reduce the cell exclusion expansion or the degree"
        )
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]          # normalized coordinates
    scale = float(np.max(np.abs(image[fit_region]))) or 1.0             # value scaling for conditioning
    A = _design_matrix(x[fit_region], y[fit_region], degree)
    z = image[fit_region] / scale
    keep = np.ones(z.size, dtype=bool)
    fitted = keep
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        fitted = keep                                                   # the set this round's fit actually uses
        coeffs = np.linalg.lstsq(A[fitted], z[fitted], rcond=None)[0]
        resid = z - A @ coeffs                                          # residuals over the WHOLE fit region
        centre = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - centre))              # robust scale (MAD), not std
        new_keep = np.abs(resid - centre) < k * sigma                    # two-sided
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    surface = (_design_matrix(x.ravel(), y.ravel(), degree) @ coeffs).reshape(h, w) * scale
    return surface, {"kept_fraction": float(fitted.sum() / background_count), "rounds": int(rounds)}


def correct_background(image_um, mask, settings):
    """Return ``(corrected_um, surface_um, exclusion, info)`` for one FOV.

    ``settings`` = ``{"method", "degree", "expand_pct"}``. ``method`` "none" subtracts a
    zero surface, "polynomial" the D14 surface fitted outside the exclusion region,
    "inpainting" the lab's biharmonic fill of the exclusion region smoothed with a
    Gaussian of sigma 5. ``info`` is ``{"kept_fraction", "rounds"}`` (1.0 / 0 when nothing
    is clipped).
    """
    method = settings["method"]
    if method not in BACKGROUND_METHODS:
        raise ValueError(f"Unknown background method {method!r}; expected one of {', '.join(BACKGROUND_METHODS)}")
    image = np.asarray(image_um, dtype=np.float64)
    exclusion = exclusion_region(mask, float(settings.get("expand_pct") or 0.0))
    if method == "none":
        surface = np.zeros_like(image)
        info = {"kept_fraction": 1.0, "rounds": 0}
    elif method == "polynomial":
        surface, info = _clipped_polynomial_surface(image, ~exclusion, int(settings["degree"]))
    else:
        inpainted_bg = inpaint_biharmonic(image, exclusion)               # lab's fit_inpainted_background
        surface = gaussian_filter(inpainted_bg, sigma=5)
        info = {"kept_fraction": 1.0, "rounds": 0}
    return image - surface, surface, exclusion, info


def diagnostics(corrected_um, mask, exclusion, info):
    """Background-fit and interior-negative readouts for one FOV.

    Background = finite pixels outside the exclusion region. ``bg_percentiles_after_nm`` feeds the
    per-FOV box plot in nm (p5, p25, p50, p75, p95) so no pixel data reaches the browser.
    """
    after = corrected_um[~exclusion] * 1e3
    after = after[np.isfinite(after)]
    if after.size:
        p5, p25, p50, p75, p95 = (float(v) for v in np.percentile(after, [5, 25, 50, 75, 95]))
    else:
        p5 = p25 = p50 = p75 = p95 = float("nan")
    # Erode each label separately so touching cells retain their own boundaries.
    interior = np.zeros(mask.shape, dtype=bool)
    for label, bbox in enumerate(find_objects(mask.astype(np.intp)), start=1):
        if bbox is not None:
            interior[bbox] |= binary_erosion(mask[bbox] == label, iterations=2)
    count = int(interior.sum())
    neg_interior_pct = float(100.0 * np.count_nonzero(corrected_um[interior] < 0) / count) if count else float("nan")

    return {
        "bg_percentiles_after_nm": (p5, p25, p50, p75, p95),
        "kept_fraction": float(info.get("kept_fraction", 1.0)),
        "rounds": int(info.get("rounds", 0)),
        "neg_interior_pct": neg_interior_pct,
    }


def line_scan(raw_um, corrected_um, surface_um, mask, label, pixel_size_um, alpha_um3_per_pg):
    """Profiles along the cell's major axis in image (row, column) coordinates.

    DIFFERS FROM LAB: use a positive sine for the column displacement; the lab's
    negative sign reflects the scan away from the major axis of rotated cells.

    Positions are pixel indices along the line; OPD profiles in nm; ``mass_pg`` is the
    corrected profile times ``mass_factor`` (DIFFERS FROM LAB: signed, micrometre based);
    ``inside`` marks samples that fall on the cell's own label.
    """
    region = next((r for r in regionprops(mask) if r.label == label), None)
    if region is None:
        raise ValueError(f"Label {label} is not in the mask")
    y0, x0 = region.centroid
    orientation = region.orientation
    length = region.major_axis_length / 2
    dy = math.cos(orientation) * length
    dx = math.sin(orientation) * length
    src = (y0 - dy, x0 - dx)
    dst = (y0 + dy, x0 + dx)

    def sample(image, order=1):
        return profile_line(image, src, dst, order=order, mode="constant", cval=0)

    corrected = sample(corrected_um)
    return {
        "position_px": np.arange(corrected.size),
        "raw_nm": sample(raw_um) * 1e3,
        "surface_nm": sample(surface_um) * 1e3,
        "corrected_nm": corrected * 1e3,
        "mass_pg": corrected * mass_factor(pixel_size_um, alpha_um3_per_pg),
        "inside": sample((mask == label).astype(np.float64), order=0) > 0.5,
    }


def cell_features(corrected_um, mask, pixel_size_um, alpha_um3_per_pg):
    """Per-cell Dry-mass statistics and Spatial texture (a trimmed copy of the lab's compute_cell_stats).

    Returns ``{label: {suffix: value}}`` over DRY_MASS_SUFFIXES + SPATIAL_TEXTURE_SUFFIXES.
    Pixels are selected by label coordinates, never by multiplying with the mask.
    DIFFERS FROM LAB: signed pixel masses (no ``np.abs``); clipping to zero only where a
    non-negative distribution is required (entropy pair, core-to-edge ratio); undefined
    small-sample statistics are NaN instead of 0 / 1; density per um^2 instead of per pixel.
    """
    factor = mass_factor(pixel_size_um, alpha_um3_per_pg)
    pixel_area_um2 = float(pixel_size_um) ** 2
    features = {}
    for region in regionprops(mask, intensity_image=corrected_um):
        coords = region.coords
        m = corrected_um[coords[:, 0], coords[:, 1]] * factor            # signed pg per pixel
        n = m.size
        total = float(m.sum())
        f = {"dry_mass_pg": total, "mass_density_pg_per_um2": total / (n * pixel_area_um2)}
        f["dry_mass_variance"] = float(m.var(ddof=1)) if n >= 2 else np.nan
        f["dry_mass_skewness"] = float(skew(m, bias=False)) if n >= 3 else np.nan
        f["dry_mass_kurtosis"] = float(kurtosis(m, fisher=True, bias=False)) if n >= 4 else np.nan
        positive = np.clip(m, 0, None)
        positive_total = positive.sum()
        if n >= 2 and positive_total > 0:
            raw_entropy = float(entropy(positive / positive_total))
            f["dry_mass_entropy"] = raw_entropy
            f["dry_mass_evenness"] = raw_entropy / math.log(n)
        else:
            f["dry_mass_entropy"] = f["dry_mass_evenness"] = np.nan
        # Spatial texture: gradient over the bounding-box crop, radial core/periphery ratio, polar offset.
        minr, minc, maxr, maxc = region.bbox
        crop = corrected_um[minr:maxr, minc:maxc] * factor
        in_cell = mask[minr:maxr, minc:maxc] == region.label
        f["avg_dm_gradient_mag"] = float(np.mean(scharr(crop)[in_cell]))
        cy, cx = region.centroid
        distances = np.hypot(coords[:, 0] - cy, coords[:, 1] - cx)
        max_distance = distances.max()
        normalized = distances / max_distance if max_distance > 0 else np.zeros_like(distances)
        core = normalized <= 0.33
        periphery = normalized >= 0.66
        periphery_mean = float(positive[periphery].mean()) if periphery.any() else np.nan
        if core.any() and periphery_mean > 0:
            f["radial_mass_index"] = float(positive[core].mean() / periphery_mean)
        else:
            f["radial_mass_index"] = np.nan
        if total > 0:
            wy, wx = region.centroid_weighted
            f["polar_gradient_index"] = float(np.hypot(cy - wy, cx - wx) / region.equivalent_diameter_area)
        else:
            f["polar_gradient_index"] = np.nan
        features[int(region.label)] = f
    return features
