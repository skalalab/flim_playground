# Intensity-only example: MCF-7 cells (BBBC021 subset)

20 fields of view (FOVs), 1,623 cells: five treatments, four sites each. Images are uint16
(widefield, 20x, 0.496 µm/px). Each FOV has `_actin_DNA_stack.tif` (two channels: 1 = F-actin,
2 = DNA), `_tubulin.tif` (β-tubulin, one 2D image) and two Cellpose-SAM label masks:
`_cell_mask.tif` (whole cells) and `_nuclei_mask.tif` (each cell's nuclei, with the cell's label).

FOVs are named `{compound}_{dose}_rep1_{well}_{site}`. Treatments: DMSO (control), taxol and
nocodazole (microtubule drugs), cytochalasin B (actin disruptor) and AZ-A (Aurora kinase inhibitor).

## FLIM Playground settings

Three `Intensity-only` channels; cell ID `cell_id`; FOV column `image_name`; categorical columns
`treatment`, `well` and `site`.

| Channel   | Intensity (2D)         | Mask               | Features            |
| --------- | ---------------------- | ------------------ | ------------------- |
| DAPI      | `_actin_DNA_stack.tif` | `_nuclei_mask.tif` | morphology, texture |
| FITC      | `_tubulin.tif`         | `_cell_mask.tif`   | morphology, texture |
| Rhodamine | `_actin_DNA_stack.tif` | `_cell_mask.tif`   | texture             |

In Data Extraction, paste this folder's path; all 20 FOVs should show ✅. Select channel 2 for DAPI
and channel 1 for Rhodamine. In the categorical step, choose `treatment`, `well` and `site`, then
pick a slot for each. The slot pickers list one FOV's values, not slot numbers. For example, in
`taxol_3uM_rep1_D03_s4`:

| Slot | Example | Meaning                                      | Map to      |
| ---- | ------- | -------------------------------------------- | ----------- |
| 1    | `taxol` | drug; DMSO is the untreated control          | `treatment` |
| 2    | `3uM`   | concentration; one per drug here             | skip        |
| 3    | `rep1`  | replicate 1 (plate 22123); same in every FOV | skip        |
| 4    | `D03`   | 96-well plate position: row D, column 3      | `well`      |
| 5    | `s4`    | site: which of the well's 4 images           | `site`      |

Each drug is in one well, so `well` groups exactly like `treatment`. The four sites are images of
one well: they show consistency across images, not independent replicates.

## Source

Images: BBBC021v1, week 1, plate 22123 (https://bbbc.broadinstitute.org/BBBC021), pixel values
unchanged; DNA and actin are merged into one file. © AstraZeneca; CC BY 4.0 on IDR (idr0035).
Please write: "We used image set BBBC021v1 [Caie et al., Molecular Cancer Therapeutics, 2010],
available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]."

Masks: Cellpose-SAM (Pachitariu et al., bioRxiv 2025, https://doi.org/10.1101/2025.04.28.651001).
