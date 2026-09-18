"""Synthetic QPI and FLIM (PTU) fields of view plus config builders for the QPI tests.

Not a test module (no ``test_`` prefix). Import as ``from qpi_fixtures import ...``: pytest puts
``tests/`` on ``sys.path`` for sibling imports, as ``test_ptu_reference_workflow`` already relies on.
A QPI FOV is a tilted plane 40 nm below zero with two flat 200 nm cells (256 and 400 pixels) and
2 nm noise, stored as float32 METRES like the fixture; a FLIM FOV is a PTU decay on a small grid.
"""
import numpy as np
import pandas as pd
import tifffile
import toml
from test_ptu_references import write_reference

from src import config
from src.widgets import metadata_widgets as mw

QPI_INPUT = "QPI (2D)"
FLIM_INPUT = "Decay (3/4D)"
QPI_EXTRACTORS = ["Intensity morphology", "Dry-mass statistics", "Spatial texture"]
ALL_EXTRACTORS = ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture",
                  "Dry-mass statistics", "Spatial texture"]
CONSTANTS = {"pixel_size_um": 0.275, "opd_unit": "m", "alpha_um3_per_pg": 0.181818}
# Cell 1: 256 px x 0.2 um x 0.275^2 / 0.181818 = 21.296 pg
CELL1_MASS_PG = 256 * 0.2 * 0.275**2 / 0.181818


def write_qpi_fov(folder, name, *, shape=(64, 64), image_suffix="_WaveFront.tiff",
                  mask_suffix="_WaveFront_cellpose.tiff", offset_nm=-40.0, cell_nm=200.0, seed=0):
    """Write ``{name}{image_suffix}`` (float32 metres) and ``{name}{mask_suffix}`` (uint16 labels).

    Returns ``(wavefront_path, mask_path, mask)``. Cell 1 is 16x16 near the top-left, cell 2
    is 20x20 past the centre; ``shape`` must be at least 48x48.
    """
    h, w = shape
    yy, xx = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    mask = np.zeros(shape, dtype=np.uint16)
    mask[h // 8: h // 8 + 16, w // 8: w // 8 + 16] = 1
    mask[h // 2 + 4: h // 2 + 24, w // 2 + 4: w // 2 + 24] = 2
    image_um = (offset_nm + 20.0 * xx - 10.0 * yy) * 1e-3 + np.where(mask > 0, cell_nm * 1e-3, 0.0)
    image_um = image_um + np.random.default_rng(seed).normal(0, 0.002, shape)
    wavefront = folder / f"{name}{image_suffix}"
    mask_path = folder / f"{name}{mask_suffix}"
    tifffile.imwrite(wavefront, (image_um * 1e-6).astype(np.float32))
    tifffile.imwrite(mask_path, mask)
    return wavefront, mask_path, mask


def write_flim_fov(folder, name, *, grid=(2, 3), decay_suffix=".ptu", mask_suffix="_mask.tif"):
    """A PTU decay with a few photons on ``grid`` plus an all-ones mask (one cell, label 1)."""
    gy, gx = grid
    data = np.zeros((3, gy, gx, 1, 32), dtype=np.uint16)
    data[0, 0, 0, 0, 3] = 7
    data[1, gy - 1, gx - 1, 0, 4] = 11
    data[2, 0, gx - 1, 0, 5] = 13
    write_reference(folder / f"{name}{decay_suffix}", data)
    tifffile.imwrite(folder / f"{name}{mask_suffix}", np.ones(grid, dtype=np.uint8))


def write_irf(folder, name="quenched.ptu"):
    """The dataset-global IRF, matched by suffix alone."""
    write_reference(folder / name)


def qpi_channel(name="QPI", *, image_suffix="_WaveFront.tiff", mask_suffix="_WaveFront_cellpose.tiff",
                extractors=QPI_EXTRACTORS, constants=CONSTANTS):
    return {"channel_name": name, "imaging_modality": "QPI", "input_type": QPI_INPUT,
            QPI_INPUT: {"selected_feature_extractors": list(extractors), **constants,
                        "input_suffixes": {QPI_INPUT: image_suffix, "Mask": mask_suffix}}}


def flim_channel(name="ch1", *, decay_suffix=".ptu", mask_suffix="_mask.tif", irf="quenched.ptu"):
    """A fit-free FLIM channel calibrated by an IRF, requiring a shift."""
    return {"channel_name": name, "imaging_modality": "FLIM", "input_type": FLIM_INPUT,
            FLIM_INPUT: {"selected_feature_extractors": ["Lifetime fit free"],
                         "input_suffixes": {"Decay": decay_suffix, "IRF": irf, "Mask": mask_suffix}}}


def intensity_channel(name="red", *, image_suffix="_WaveFront.tiff", mask_suffix="_WaveFront_cellpose.tiff"):
    return {"channel_name": name, "imaging_modality": "Intensity-only", "input_type": "Intensity (2D)",
            "Intensity (2D)": {"selected_feature_extractors": ["Intensity morphology"],
                               "input_suffixes": {"Intensity (2D)": image_suffix, "Mask": mask_suffix}}}


def write_config(tmp_path, monkeypatch, channels):
    """Point src.config at a profile holding ``channels`` (from the builders above); clear file caches."""
    cfg = {"num_channels": len(channels), "flim_decay_input_type": FLIM_INPUT,
           "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
           "all_feature_extractors": list(ALL_EXTRACTORS),
           FLIM_INPUT: {"available_feature_extractors": ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"],
                        "file_types": ["Decay", "IRF", "Mask"], "laser_rate": 0.04, "fit_free_calibration": "IRF"},
           "Intensity (2D)": {"available_feature_extractors": ["Intensity morphology", "Intensity texture"],
                              "file_types": ["Intensity (2D)", "Mask"]},
           QPI_INPUT: {"available_feature_extractors": list(QPI_EXTRACTORS), "file_types": [QPI_INPUT, "Mask"]}}
    for i, channel in enumerate(channels, 1):
        cfg[f"ch{i}"] = channel
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"current_profile": "default", "profiles": {"default": cfg}}), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    mw.clear_folder_scan_caches()
    return path


def qpi_metadata_rows(folder, names, *, channel="QPI", recipe=None, extractors=QPI_EXTRACTORS,
                      image_suffix="_WaveFront.tiff", mask_suffix="_WaveFront_cellpose.tiff"):
    """Source rows for a QPI channel, optionally with a confirmed background recipe."""
    rows = []
    for name in names:
        row = {"image_name": name,
               f"{channel}_{QPI_INPUT}": str(folder / f"{name}{image_suffix}"),
               f"{channel}_Mask": str(folder / f"{name}{mask_suffix}"),
               f"{channel}_input_type": QPI_INPUT, f"{channel}_imaging_modality": "QPI",
               "derived_features": "[]"}
        for extractor in extractors:
            row[f"{channel}_{extractor}"] = True
        for key, value in CONSTANTS.items():
            row[f"{channel}_{key}"] = value
        if recipe is not None:
            row[f"{channel}_bg_method"] = recipe["method"]
            row[f"{channel}_bg_degree"] = recipe["degree"]
            row[f"{channel}_bg_expand_pct"] = recipe["expand_pct"]
        rows.append(row)
    return pd.DataFrame(rows)
