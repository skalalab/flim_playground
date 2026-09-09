# Data Extraction examples

## Mosaic NADH images

[Download the Mosaic example ZIP](https://raw.githubusercontent.com/skalalab/flim_playground/main/example_data/Data_Extraction/3d_decay_1channel.zip) (9.7 MB), then unzip it and use the `3d_decay_1channel` folder in FLIM Playground.

The archive contains 25 SDT images (`Mosaic01.sdt` through `Mosaic25.sdt`), their 25 matching cell-mask TIFFs, and the shared `nadh_irf.txt`. It also includes a README with the extraction settings and a `SHA256SUMS` inventory of all 51 input files. Images have 512 x 512 pixels, 256 time bins, and a 12.5 ns period. Configure one NADH channel, `Decay (3/4D)`, `Lifetime fit` and `Lifetime fit free`, and IRF calibration. Use `.sdt`, `_Ch2_summed_cellpose.tiff`, and `nadh_irf.txt` as the decay, mask, and IRF suffixes.

See the manual's [Numerical Feature Extraction chapter](https://skalalab.github.io/flim_playground_doc/numerical_feature_extraction.html) for more detail on extraction and calibration. The raw files and masks are unchanged from the `3d_decay_1channel` example in the FLIM Playground dataset collection; optional SPCImage pre-fitted outputs are not included.

Archive SHA-256:

```text
254536e95b5b702d5b8acbb31c9ee73348d1d1a3a85b2af950853846ecca09b0
```

## T-cell activation decays

[T_cell_activation](T_cell_activation/) contains the 2D cell-decay CSVs and IRF used in the T-cell extraction demo. See the [repository README](../../README.md#data-extraction-demo) for the demonstration and source paper.

## Citation

Zhao, W., Samimi, K., Skala, M. C., & Datta, R. (2026). *Example and validation datasets for FLIM Playground* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19774943
