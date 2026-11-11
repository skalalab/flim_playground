# Data Extraction examples

Each folder is a complete input set for one kind of experiment. To run one:

1. On **Home**, create a profile for the example (for example `2d_decay`), enter the settings
   listed below, and click **Update Configuration**. A separate profile per example keeps one
   example's settings from overwriting another's.
2. Open **Data Extraction → Numerical**, paste the folder's full path, and check that every FOV
   shows ✅. Then calibrate and extract. The feature table is saved in the same folder.
3. Optionally, open **Data Extraction → Categorical** to add labels such as treatment, parsed from
   the FOV names.

These settings are the same for every example:

- Keep the defaults **Unique cell identifier column name** = `cell_id` and **FOV column name** =
  `image_name`.
- Files are paired by name: each file of a FOV is named `<FOV name><suffix>`. Files in subfolders
  are found too.
- The app reads image size and timing from `.sdt` files. Only `Decay (2D)` needs them typed in.

| Folder                                                   | What it shows                                    | FLIM input format                | Channels             | FOVs | Cells |
| -------------------------------------------------------- | ------------------------------------------------ | -------------------------------- | -------------------- | ---- | ----- |
| [`2d_decay`](#2d_decay)                                  | Decays already summed per cell, no images        | `Decay (2D)`                     | 1 FLIM               | 3    | 796   |
| [`3d_decay_1channel`](#3d_decay_1channel)                | One `.sdt` per FOV; SPCImage pre-fitted variant  | `Decay (3/4D)` or pixel-prefitted | 1 FLIM              | 25   | 584   |
| [`4d_decay_1channel`](#4d_decay_1channel)                | One `.sdt` holding three detectors               | `Decay (3/4D)`                   | 1 FLIM               | 1    | 176   |
| [`flim_qpi`](#flim_qpi)                                  | Two FLIM channels plus quantitative phase (QPI)  | `Decay (3/4D)`                   | 2 FLIM + 1 QPI       | 4    | 282   |
| [`intensity_3channel_stack`](#intensity_3channel_stack)  | Intensity images only, one two-plane TIFF        | none                             | 3 Intensity-only     | 20   | 1,623 |
| [`categorical_extraction`](#categorical_extraction)      | Finished feature tables for the Categorical step | none                             | not used             | 27   | 1,281 |

The manual covers each step in more detail:
[Numerical Feature Extraction](https://skalalab.github.io/flim_playground_doc/numerical_feature_extraction.html)
and
[Categorical Feature Extraction](https://skalalab.github.io/flim_playground_doc/categorical_feature_extraction.html).

## 2d_decay

T-cell NAD(P)H decays supplied as tables instead of images. There are three FOVs: `Tcell_Act`
(activated), `Tcell_Qui` (quiescent) and `Tcell_bckgrnd` (background). In each
`<FOV>_filtered.csv`, a row is one cell's decay, already summed over the cell, and the 200
columns are time bins. `IRF.txt` is the instrument response (200 values) shared by all three.
There is no mask, so morphology features are unavailable.

| Setting                                   | Value                                                     |
| ----------------------------------------- | --------------------------------------------------------- |
| Number of channels                        | `1`                                                       |
| Channel 1 name / Imaging modality         | `NADH` / `FLIM`                                           |
| FLIM input format                         | `Decay (2D)`                                              |
| Decay (2D) duration (ns) / time bins      | `20` / `200` (time bins must equal the CSV's column count) |
| Extract feature types                     | `Lifetime fit`, `Lifetime fit free` (`Intensity texture` optional) |
| Number of components                      | `2`                                                       |
| Laser rate (MHz)                          | `50`                                                      |
| Fit free calibration method               | `IRF`                                                     |
| Suffixes: Decay / IRF                     | `_filtered.csv` / `IRF.txt`                               |
| Categorical columns                       | add `condition`                                           |

**Categorical step:** `Tcell_Act` splits into `Tcell` | `Act`. Map `condition` to the second
slot.

## 3d_decay_1channel

Mosaic NAD(P)H images: 25 FOVs, `Mosaic01`–`Mosaic25`. Each FOV has a `.sdt` decay (512 × 512
pixels, 256 time bins, 12.5 ns) and a Cellpose cell mask, `_Ch2_summed_cellpose.tiff`.
`nadh_irf.txt` is the IRF shared by all 25. This is the example used in the manual's
[quick start](https://skalalab.github.io/flim_playground_doc/#try-data-extraction-locally) and
[IRF calibration walkthrough](https://skalalab.github.io/flim_playground_doc/irf_calibration.html#time-gates).

| Setting                           | Value                                   |
| --------------------------------- | --------------------------------------- |
| Number of channels                | `1`                                     |
| Channel 1 name / Imaging modality | `NADH` / `FLIM`                         |
| FLIM input format                 | `Decay (3/4D)`                          |
| Extract feature types             | `Lifetime fit`, `Lifetime fit free`     |
| Number of components              | `2`                                     |
| Laser rate (MHz)                  | `80`                                    |
| Fit free calibration method       | `IRF`                                   |
| Suffixes: Decay / Mask / IRF      | `.sdt` / `_Ch2_summed_cellpose.tiff` / `nadh_irf.txt` |

### Pixel-prefitted variant: `SPCImage_prefitted/`

`SPCImage_prefitted/` holds per-pixel two-component SPCImage fits of the same 25 FOVs. Each FOV
has `_t1.asc`, `_t2.asc` and the amplitude fractions `_a1[%].asc` and `_a2[%].asc`. The plain
`_a1.asc` and `_a2.asc` files are amplitudes in counts, and Mosaic16's `_chi`, `_offset` and
`_shift` maps are extra; the app reads none of them. Instead of fitting, the app averages each
map over every cell mask. Starting from the settings above, change:

| Setting                       | Value                                                                    |
| ----------------------------- | ------------------------------------------------------------------------ |
| FLIM input format             | `Decay (3/4D) pixel-prefitted`                                           |
| Extract feature types         | `Lifetime fit` (add `Lifetime fit free` to also compute phasors from the `.sdt`) |
| Suffixes: Mask / SPCImage t1  | `_Ch2_summed_cellpose.tiff` / `_Ch2_summed_t1.asc`                       |
| Suffixes: Decay / IRF         | `.sdt` / `nadh_irf.txt`, only asked for with `Lifetime fit free`         |

You enter only the t1 suffix. The app derives the others by replacing `_t1.asc` with `_t2.asc`
and `_a1[%].asc`.

## 4d_decay_1channel

One FOV, `0hr_imm_1n`, recorded on three detectors in a single `.sdt` (256 × 256 pixels, 256 time
bins, 10 ns). NAD(P)H is on detector 3: detector 1 is empty and detector 2 is dim. The mask
`N_Masks/0hr_imm_1n_photons_cell_masks.tif` sits in a subfolder, and `220913_nadh.csv` is the
IRF.

| Setting                           | Value                                                      |
| --------------------------------- | ---------------------------------------------------------- |
| Number of channels                | `1`                                                        |
| Channel 1 name / Imaging modality | `NADH` / `FLIM`                                            |
| FLIM input format                 | `Decay (3/4D)`                                             |
| Extract feature types             | `Lifetime fit`, `Lifetime fit free`                        |
| Number of components              | `2`                                                        |
| Laser rate (MHz)                  | `80`                                                       |
| Fit free calibration method       | `IRF`                                                      |
| Suffixes: Decay / Mask / IRF      | `.sdt` / `_photons_cell_masks.tif` / `220913_nadh.csv`     |

**Data Extraction:** after the folder scan, choose detector **3** in the NADH channel picker. With
only one FOV there is nothing to group, so the Categorical step is optional.

## flim_qpi

Multimodal imaging of the same cells: FAD and NADH lifetimes plus quantitative phase (dry mass).
There are four FOVs: `day_treatment1_1`, `day_treatment1_2`, `day_treatment2_1` and
`day_treatment2_2`. Each FOV has five files:

| File                        | Contents                                                                 |
| --------------------------- | ------------------------------------------------------------------------ |
| `<FOV>f.sdt`                | FAD decays; signal on detector 2 only                                    |
| `<FOV>n.sdt`                | NADH decays; signal on detectors 1 and 3                                 |
| `<FOV>n_cellpose.tiff`      | Cell mask for both FLIM channels (256 × 256)                             |
| `<FOV>_WaveFront.tiff`      | QPI optical path difference, in metres (552 × 552)                       |
| `<FOV>_WaveFront_cellpose.tiff` | Cell mask for QPI; same labels as the FLIM mask, so cells match across modalities |

The IRFs are `Ch2_IRF_890__2024.txt` for FAD and `Ch3_IRF_750__2024.txt` for NADH.
`Ch1_IRF_890__2024.txt` is not used by this setup.

**Channels** (Number of channels = `3`):

| Channel | Name   | Imaging modality | Extract feature types                                                         |
| ------- | ------ | ---------------- | ----------------------------------------------------------------------------- |
| 1       | `QPI`  | `QPI`            | `Intensity morphology`, `Dry-mass statistics`, `Spatial texture`              |
| 2       | `FAD`  | `FLIM`           | `Lifetime fit`, `Lifetime fit free`, `Intensity morphology`, `Intensity texture` |
| 3       | `NADH` | `FLIM`           | `Lifetime fit`, `Lifetime fit free`, `Intensity morphology`, `Intensity texture` |

**Shared FLIM settings:** FLIM input format `Decay (3/4D)`, laser rate `80` MHz, fit free
calibration `IRF`, `2` components for both FAD and NADH.

**QPI constants:** pixel size `0.275` µm, OPD unit in file `m`, α `0.181818` µm³/pg (the default).

**Suffixes.** These start without `_` because the FOV name ends right before them
(`day_treatment1_1` + `f.sdt`).

| Channel | Decay or QPI (2D) | Mask                       | IRF                     |
| ------- | ----------------- | -------------------------- | ----------------------- |
| QPI     | `_WaveFront.tiff` | `_WaveFront_cellpose.tiff` |                         |
| FAD     | `f.sdt`           | `n_cellpose.tiff`          | `Ch2_IRF_890__2024.txt` |
| NADH    | `n.sdt`           | `n_cellpose.tiff`          | `Ch3_IRF_750__2024.txt` |

Add `treatment` to **Categorical columns**.

**Data Extraction:** after the folder scan, choose detector **3** for NADH, the detector that
matches `Ch3_IRF_750__2024.txt`. FAD has only one detector with signal, so the app picks it
automatically. Calibration has two parts: IRF shifts for FAD and NADH, and background correction
for QPI. Confirm both before extracting.

**Categorical step:** `day_treatment1_1` splits into `day` | `treatment1` | `1`. Map `treatment`
to the second slot. The first slot is `day` in every FOV and the third numbers the FOV, so skip
both.

## intensity_3channel_stack

MCF-7 cells from BBBC021, 20 FOVs: five drug treatments with four sites each. There is no FLIM.
The DNA and actin channels share one two-plane TIFF, and each channel picks its plane. Full
details and the image source are in [this folder's README](intensity_3channel_stack/README.md).
In short:

| Channel   | Imaging modality | Intensity (2D) suffix  | Mask suffix        | Plane | Extract feature types                       |
| --------- | ---------------- | ---------------------- | ------------------ | ----- | ------------------------------------------- |
| DAPI      | `Intensity-only` | `_actin_DNA_stack.tif` | `_nuclei_mask.tif` | 2     | `Intensity morphology`, `Intensity texture` |
| FITC      | `Intensity-only` | `_tubulin.tif`         | `_cell_mask.tif`   |       | `Intensity morphology`, `Intensity texture` |
| Rhodamine | `Intensity-only` | `_actin_DNA_stack.tif` | `_cell_mask.tif`   | 1     | `Intensity texture`                         |

Categorical columns: `treatment`, `well` and `site` (slots 1, 4 and 5 of
`taxol_3uM_rep1_D03_s4`).

## categorical_extraction

Nine `single_cell_features_<timestamp>.csv` tables are already extracted by the Numerical step,
one per day × treatment of a QPI + FAD + NADH experiment (the `flim_qpi` channel setup). Together
they cover three days, three treatments and three FOVs each. There are no raw images, so use only
**Data Extraction → Categorical**.

Profile: add `day` and `treatment` to **Categorical columns**. Channel settings do not matter for
this step.

1. In **Categorical**, paste this folder's path and keep the delimiter `_`. All nine tables are
   found and merged; columns that some tables lack are dropped and listed.
2. Choose `day` and `treatment`. For the FOV `day1_treatment1_1`, map `day` to slot 1 (`day1`)
   and `treatment` to slot 2 (`treatment1`). Skip slot 3, the FOV number.
3. Click **Confirm category mapping & export the combined dataset**. The labeled table is saved
   here as `<timestamp>_combined.csv`, ready for Data Analysis. Later scans skip `_combined.csv`
   files.

## Citation

Except for `intensity_3channel_stack`, which credits its own source, please cite:

Zhao, W., Samimi, K., Skala, M. C., & Datta, R. (2026). *Example and validation datasets for
FLIM Playground* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19774943
