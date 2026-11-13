# Signed whole-brain effective connectivity and shared-variance removal

Code accompanying the manuscript:

> Wu, H., Tang, J., Wang, W. *Negative weights in signed generative effective connectivity from resting-state fMRI track shared-variance removal and are not specific to competition.* Manuscript submitted.

The repository contains:
- an independent reimplementation of the signed generative effective connectivity (GEC) estimator of Luppi et al. (2026, *Nature Neuroscience*);
- the preprocessing, simulation and analysis code for every result in the paper;
- the registration scripts that recorded each analysis specification;
- the rendered figures and the summary tables underlying every figure.

## Repository layout

```
code/         analysis code (Python; three MATLAB wrappers), logically identical to the versions frozen and executed for the paper (see "Provenance")
figures/      figdata/ (summary tables underlying every figure), output/ (rendered figures, PDF and PNG)
provenance/   fingerprints.json: frozen SHA-256, released SHA-256 and logic fingerprint of every script;
              records.md: SHA-256 prefixes of the registration and freeze records of every analysis;
              frozen_laws/: the frozen readout laws g, g_nf and g_sim at full precision
tools/        verify_logic.py: checks that code/ is logically identical to the frozen versions
```

## Mapping between the paper and the code

The table maps the scripts to the sections of the paper.

| Paper | Scripts in `code/` |
|---|---|
| Estimator (Section 2.2) and validation fixtures F1–F5 (Supplementary Note 2) | `luppi_hopf.py`, `luppi_r4e_lib.py`, `luppi_r4e_run.py` (the fixture script is not distributed; see Provenance); comparison with the authors' unmodified MATLAB code (Section 2.2.3, Supplementary Note 13): `vg_run.py`, `vg_analyze.py`, `vg_matlab_run.m` |
| Preprocessing comparison, PPC (Sections 2.3, 3.2) | `luppi_pipeline_vol.py`, `luppi_ppc_prep.py`, `luppi_r4e_ppc.py`, `luppi_r4e_ext.py`, `luppi_ppc_analyze.py`, `luppi_r4e_e8_analyze.py`; surface ICA-FIX condition: `luppi_r4e_e5fix.py`, `luppi_r4e_e5fix_analyze.py` |
| Participant-specific tissue masks (Sections 2.4.1, 3.2) | `b1_run.py`, `b1_analyze.py`, `b1_prereg.py`, `b1_fixture.py`, `wmparc2_etag.py` |
| In-sample model comparison (Sections 2.3.3, 3.2) | `ppc_fccorr_bridge.py` |
| Group-level FC correlation (Sections 2.3.3, 3.2), its held-out version and the ICA-FIX transport of the readout law (Sections 3.2, 3.3) | `r5_groupfc.py`, `r5bc_heldout_fix.py` |
| Participant-specific SC and 232-region parcellation (Sections 2.4.2, 3.7.1, 3.7.2; Supplementary Note 12) | SC construction and fidelity calibration: `r23_sc.py`; 232-region time series with the frozen PPC denoising: `r3_prep.py`; fits: `r23_run.py`; analysis: `r23_analyze.py`; reference-SC identity check (Section 2.1): `r2_sc_identity.py`; descriptive checks of the calibration failure: `r23_gsc_diag.py`, `r23_gsc_diag2.py`, `r23_gsc_diag3.py`, `r23_gsc_diag4.py` |
| Edge-level analysis (exploratory; Sections 2.3.4, 3.2; Supplementary Note 5), random-placement null and filter cross-terms (Sections 2.3.3, 2.3.4, 3.1.1, 3.2; Supplementary Note 5) | `e1i1_explore.py`; `vavf_desc.py` (stages `vf`, `vc`) |
| Readout law and validations V1, V2 (Sections 2.5, 2.5.1, 3.3; frozen coefficients in `provenance/frozen_laws/`) | `e12a_g_freeze.py`, `e12a_rl.py`, `e12a_validate.py`, `e12a_prereg.py`, `e12a_new.py`, `e12a_v2_select.py`, `e12a_validate_v2.py`, `e12a_prereg_v2.py`, `e12a_xcheck.py`, `e12a_v3_diag.py`; operator-only baseline (Supplementary Note 6): `vavf_desc.py` (stage `ve`) |
| Simulation 1 (Sections 2.6, 3.4) | `luppi_r4e_e10c.py`, `luppi_r4e_e10c_analyze.py`, `e10c_targets.py`, `e10c_sources.py`, `e10c_prereg.py` |
| Simulation 2 (Sections 2.6, 3.4, 3.5) | `luppi_r4e_e11.py`, `luppi_r4e_e11_analyze.py`, `e11_prereg.py` |
| Simulation 3 (Sections 2.5.2, 3.3) | `e12c_support_freeze.py`, `e12c_run.py`, `e12c_analyze.py`, `e12c_prereg.py` |
| Simulation 4 (Sections 2.5.2, 3.3) | `e12d_gsim_freeze.py`, `e12d_run.py`, `e12d_analyze.py`, `e12d_prereg.py` |
| Simulation 5, L1 control (Sections 2.8.1, 2.8.2, 3.4.2) | `e14b_run.py`, `e14b_analyze.py`, `e14b_prereg.py`, `e14b_fixture.py`. The initial run with a relative threshold (Supplementary Note 10) is in `e14_run.py` and `e14_analyze.py`; `e14_analyze.py` also provides helpers used by `e14b_analyze.py`. |
| Simulations 6 and 6b, extension of the original L1 ground-truth test (Sections 2.8.3, 2.8.4, 3.4.3; Supplementary Note 10) | Truth selection and input check: `r1_smoke.py`; Simulation 6: `r1_run.py`, `r1_analyze.py`, with the exploratory analyses `r1_explore_a0.py` (baseline vs input negative FC) and `r1_audit.py` (negatives by true weight, stopping); Simulation 6b: `r1p_run.py`, `r1p_analyze.py` |
| Tissue-pool check (Sections 2.6, 3.4.1, 3.5; Supplementary Note 7) | `vavf_vb.py` |
| Public-configuration checks (Sections 2.4.3, 3.7.3, 3.7.4; Supplementary Note 14) | released configuration (K3): `vavf_va.py`; closest public configuration (K4): `vavf_va4.py`; code registration of the checks above: `vavf_rules.py` |
| Hierarchy analysis (Sections 2.7, 3.6) | `e13_hier.py`, `e13_analyze.py`, `e13_prereg.py`, `e13_fixture.py`, `myelin_etag.py`; spin test and its paired follow-up (Supplementary Note 9): `r4_spin.py`, `r4b1_paired.py`; original hierarchy estimand O_h (Supplementary Note 9): `vavf_vd.py` |
| Simulation infrastructure: truth generation, simulator, shared analysis helpers | `luppi_r4e_e7.py`, `luppi_r4e_e8.py`, `luppi_r4e_e6.py`, `luppi_r4e_analyze.py` |
| Additional zero-fit checks (Supplementary Notes 2, 7) | `factcheck_extra.py` |
| Numerical verification of the linear equivalence (Supplementary Note 1) | `ldsec_t13_overlap.py`, `ldsec_t14_theorem.py`, `ldsec_t14b_general.py` |
| Distributional comparison of the two implementations (Section 2.2.3; Supplementary Note 13) | `vh_run.py`, `vh_matlab_run.m`; code registration: `vhvi_rules.py` |
| Frequency control (Sections 2.3.2, 3.2; Supplementary Note 15) | `vi2_run.py` |
| Bifurcation parameter a = −0.025 and distributional reproduction check (Sections 2.2.3, 2.4.3, 3.7.5; Supplementary Notes 13, 14) | authors' code on the example data: `vj1_run.py`, `vj_matlab_run.m`; configuration checks K1a and K4a: `vj2_run.py`; K3a, added by the prelisted selection rule: `vj2b_run.py`; code registration: `vj_rules.py` |
| Data root (added for the release) | `paths.py` |

## Requirements

- Python 3.12 with NumPy 2.5, SciPy 1.18, JAX 0.11 (64-bit mode is enabled in the code) and nibabel 5.4; see `environment.yml`.
- DSI Studio 2025.04.16 (Windows CPU build) for tractography. The scripts expect it at `tools/dsi_studio_2025.04.16/dsi_studio_win/dsi_studio.exe` under the data root.
- MATLAB R2023b with the Signal Processing and Statistics toolboxes, only for the comparisons with the authors' code (`vg_run.py`, `vh_run.py`, `vj1_run.py`); set `SGEC_MATLAB` to the MATLAB executable if it is not on the PATH.
- The fits were run on two Windows workstations with 64 logical cores each. A full replication of all experiments requires on the order of 10⁴ CPU-hours.

## Data

The input data are not redistributed here; the derived summary tables in `figures/figdata/` are described under "License".
- **HCP.** Human Connectome Project S1200 data, under the HCP data use terms (ConnectomeDB, now hosted by BALSA), for the participants listed in the paper:
  - minimally preprocessed volumetric `rfMRI_REST1_LR/RL`, run brain masks and movement regressors;
  - `MNINonLinear/ROIs/wmparc.2.nii.gz` and `MNINonLinear/ROIs/Atlas_wmparc.2.nii.gz`;
  - `MyelinMap_BC_MSMAll` surfaces;
  - the fs_LR 32k sphere surfaces of one participant (100307), for the spin test;
  - the HCP tutorial group myelin map.
- **Original authors' materials.** The example data and structural connectivity distributed with their code (github.com/Hana-Ali/competitive-cooperative-hopf). The implementation check expects the archive of commit da592aa unpacked at `raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/` under the data root; with `SGEC_DATA_ROOT` set, it must end with a slash, because the MATLAB wrapper reads it directly.
- **Atlases.**
  - The Schaefer 2018 100-parcel, 7-network atlas (volumetric FSL MNI152 2 mm and fs_LR dlabel).
  - For the 232-region analysis: the Schaefer 2018 200-parcel, 7-network atlas (FSL MNI152 2 mm) and the Tian S2 3T subcortical atlas.
  - For one descriptive check only: the Schaefer 100-parcel atlas in MNI152NLin2009cAsym space from TemplateFlow.
- **Diffusion data.** HCP QSDR reconstructions (`{id}.qsdr.fz`) distributed by the DSI Studio Fiber Data Hub (github.com/data-hcp/lifespan, release `hcp-ya`). For one descriptive check, the raw `{id}.sz` files of the three example participants were also used.

## Paths and execution

- **Data root.** All file locations are relative to a single data root, `DATA_ROOT`, defined in `code/paths.py`. Set the environment variable `SGEC_DATA_ROOT` to your data directory. If it is unset, `workspace/` in the repository root is used. Under the root, inputs go in `raw/`, intermediate time series in `work/`, and outputs and records in `results/p33/`. The scripts create the subdirectories they write to; the directory and file names expected under `raw/` are given in the scripts that read them.
- **Working directory.** Run the scripts from `code/`. They record the SHA-256 of sibling scripts by file name.
- **Experiment stages.** Each experiment is registered and frozen before it is run and analysed. The stage is selected by the first command-line argument of each script, and a stage refuses to start if the records it depends on are missing or have changed. The MATLAB wrappers `vg_matlab_run.m`, `vh_matlab_run.m` and `vj_matlab_run.m` are called by `vg_run.py`, `vh_run.py` and `vj1_run.py` from `scripts/vg/`, `scripts/vh/` and `scripts/vj/` under the data root, where they must be placed.
- **Integrity checks.** Freeze records pin the SHA-256 of all inputs and code. Run and analysis stages refuse to proceed if anything has changed.
- **Specification documents.** `r1_run.py`, `r1p_run.py`, `r23_run.py`, `vg_run.py`, `vavf_rules.py`, `vavf_va4.py`, `vhvi_rules.py`, `vi2_run.py`, `vj_rules.py` and `vj2_run.py` also hash their specification documents, expected under `docs_r1/` in the data root (file names in the scripts). The specification documents are not part of this repository. To rerun, place your own specification at that path before registering the scripts (stage `rules`).

## Provenance

**Terminology.** "Registration" in file names and records refers to locally recorded, hash-stamped specification records (Supplementary Note 11).

`provenance/records.md` lists, for each analysis of the paper, the SHA-256 prefixes of its registration records (specification, decision rules and scripts) and freeze record (pinned inputs), and the number of fits (Supplementary Note 11).

`provenance/frozen_laws/` contains the frozen readout laws at full precision, as written by the freeze scripts (Supplementary Note 6):
- `e12a_g.json` (the readout-law entry of `records.md`): the human-trained law g and the nf-only comparator g_nf, with the training ranges and the hash of the training table. Coefficients are on standardised predictors (polynomial terms in the order intercept, nf, nf², nf³, gs, gs², gs³ for g); the means (`mu`) and standard deviations (`sd`) used for standardisation are included. Supplementary Note 6 gives the same laws on the raw scale to four significant figures.
- `e12d_gsim.json` (the g_sim entry of `records.md`): the in-family law g_sim of Simulation 4, a cubic in nf on the raw scale, with the hashes of the source freeze records.

`provenance/fingerprints.json` lists three values for each script (Python and MATLAB):
- the SHA-256 of the frozen original, as recorded in the registration and freeze records;
- the SHA-256 of the released copy;
- a logic fingerprint: for Python, the SHA-256 of the abstract syntax tree after docstrings are removed, the values of the metadata fields `note`, `spec` and `stage` are blanked, and every path under the data root and the MATLAB executable are written as placeholders; for MATLAB, the SHA-256 of the text after full-line comments and blank lines are removed and the data root is written as a placeholder. The fingerprint of each frozen original was computed the same way.

All other code, including every string literal and constant, enters the fingerprint unchanged.

**Not included.** `luppi_fixtures.py`, the deterministic correspondence fixtures F1–F5 of Supplementary Note 2, contains literal NumPy transcriptions of two of the original authors' MATLAB functions, whose repository carries no licence; it is available to editors and reviewers on request. The SHA-256 of the frozen original is `24b642014e3be24bd5c340337190eab604455ed630e1fab80156d36f480f6fdb`. The `freeze` stage of `luppi_r4e_run.py` records the hash of this script and therefore requires it.

To check that every released script is logically identical to its frozen original, run:

```
python tools/verify_logic.py
```

## Figures

`figures/output/` contains Figures 1–6 and Supplementary Figures S1 and S2 as they appear in the manuscript (PDF and PNG). `figures/figdata/` contains the summary table (CSV) from which each panel was drawn; the file name gives the figure and panel:
- Figure 1: `fig1_matrices.csv`;
- Figure 2: `fig2AB_ppc_participants.csv`, `fig2C_edge_consistency.csv`, `fig2D_participant_masks.csv`, `fig2E_insample_model_comparison.csv`;
- Figure 3: `fig3_readout_ws1.csv` (training rows and accS) and `fig3_readout_ws2.csv` (V1 and V2);
- Figure 4: `fig4A_simulation1_cells.csv`, `fig4A_real_reference.csv`, `fig4B_simulation5_fits.csv`, `fig4C_tissue_pool_fits.csv`;
- Figure 5: `fig5A_simulation2_recovery.csv`, `fig5B_simulation2_compatibility.csv`;
- Figure 6: `fig6A_original_estimand.csv`, `fig6B_hierarchy_gap.csv` (set labels `sim1_`, `sim2_` and `sim3_` denote Simulations 1–3);
- Supplementary Figures S1 and S2: `figS1_robustness.csv`, `figS2_reproduction.csv`.

## License

The code and documentation are released under the MIT License; see `LICENSE`.

The summary tables in `figures/figdata/` that contain values computed from Human Connectome Project data, including the per-participant tables identified by HCP subject ID, are derived data. They are redistributed under the HCP Open Access Data Use Terms (https://www.humanconnectome.org/storage/app/media/data_use_terms/DataUseTerms-HCP-Open-Access-26Apr2013.pdf) and not under the MIT License. Only HCP Open Access data were used.

Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.
