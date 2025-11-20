# Project History
(The followings are not peer-reviewed and represent only my own thoughts.)
The development history of FLIM Playground is interesting in its own right and, hopefully, shares common traits with other “platform-y” projects. By jotting down a few memorable milestones here, I want to remind myself that:
- All of the architectural decisions grew out of synthesizing diverse needs by coming up with meaningful abstractions. The needs are specific, **data-driven**, and from them abstract hypothesis are constructed (**hypothesis-driven**), which will in turn place constraints on how people view things. This feedback loop is neither top-down—where a hypothesis dictates data analysis or even collection—nor bottom-up—where fragmented one-off solutions address isolated needs. Instead, it reflects a more sustained way of doing research (because loops by definition go indefinitely), which is precisely the type of research that FLIM Playground aims to facilitate. I am proud that it has been developed to try to exemplify this approach.
- Those "loops" are iterations of a product, an artwork, a decision, a process of carving certainty out of ambiguity and possibility, propelled by constraints, tipped by imbalancedness between ideal and reality, tickled by the desire to approach perfection while reminding oneself of its unreachability.

It starts from the very specific need of identifying single-cell outliers in UMAP. 

## Milestone 1 – Early Architecture (Dec 2024)
**Conceptual shift:** turn ad-hoc FLIM scripts into a deployable Streamlit workbench with navigable workflows.
- `f30a795` (2024-12-06) demonstrated the “workbench” idea by repackaging the PCA outlier script into a Streamlit page with uploads, feature selectors, and standardized PCA renders—showing that the tooling could live as a shared UI instead of throwaway notebooks.
- `8e08d2a` (2024-12-10) carved that prototype into its own page and introduced the multipage layout, proving that workflows could be isolated yet still share navigation/state within one app.
- `397446e` (2024-12-12) layered clustering, UMAP, and image-level QC into the same navigation setup, connecting multiple analytical modes under the “playground” metaphor rather than keeping them as separate demos.
- `6ac3fa3` (2024-12-15) added shared widgets/config so future workflows could plug into the same UI scaffolding, turning the codebase from “one page per idea” into a reusable application framework.

## Milestone 2 – Modular Feature System (Mar 2025)
**Conceptual shift:** Group features into three feature classes—identifiers, categorical metadata, and numeric aggregates—and let each class drive a distinct interaction path. 
- `05c3fce` (2025‑03‑25) defined that taxonomy at the data layer by introducing `required_cols`, `categorical_cols`, and grouped numeric tuples, then trimming uploaded CSVs down to exactly those buckets so every downstream workflow shared the same schema.
- `f27e610` (2025‑03‑26) turned the taxonomy into UX rules: categorical columns became filter widgets, while grouped numeric tuples powered single- and multi-select feature selectors, making the UI reflect the three-class separation.
- `b437157` (2025‑03‑27) proved the pattern across complex workflows by rebuilding classification on top of the shared dataset loaders and widgets, so classifiers now consume categorical combos for labels, and numeric selections for features without bespoke glue.

## Milestone 3 – Config Era (Jul 2025 - )
**Conceptual shift:** drive the app from configuration, linking metadata ingestion, analysis recipes, and channel-aware acquisition settings.
- `b23a1ca` (2025-07-09) elevated configuration to the front door: users now describe their channels, feature types, and suffixes before running anything, making `config.toml` the single source of truth for every workflow.
- `21931b0` (2025-07-10) rewrote the Data Extraction stages so every prompt, suffix validation, and export step is driven by `config.toml`, showing that metadata ingestion can run off declarative settings rather than hard-coded assumptions.
- `bfee8d4` (2025-07-26) mirrored the pattern on the analysis side by introducing `analysis_config.toml` so IDs, categorical columns, and numeric feature groups are defined once and reused across visualizations and classifiers.
- `40c65fc` (2025-08-20) brought reference dyes, IRFs, and new intensity features into the same config-driven flow, ensuring even calibration data followed the declarative contract instead of bespoke scripts.
- `edab183` (2025-09-03) showed the config could span modalities by letting intensity-only channels share the pipeline with FLIM channels without custom code paths.
- `ece0b53` (2025-11-15) extended the idea to multiple profiles so analysts can store and swap whole schema/feature setups, making configuration a reusable asset rather than a one-off checklist.
