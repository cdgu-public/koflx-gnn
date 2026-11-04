# koflx-gnn

Code for the paper "Evaluating graph neural networks on simulated knockout fluxomes for cancer gene essentiality prediction" (RECOMB 2027 submission, draft).

The pipeline builds tINIT context-specific genome-scale metabolic models of 59 colorectal cell lines (DepMap 24Q4) from Human-GEM v2.0.0, repairs and gap-fills them, applies medium uptake caps, computes a pFBA reference flux and a quadratic MOMA knockout fluxome for every single-gene knockout (Gurobi), re-solves knockouts at a tighter solver tolerance, assembles the knockout sample set with a growth mask, and trains graph neural networks on two graph encodings of each knockout fluxome to predict DepMap gene dependency. Evaluation uses gene-disjoint and cell-line-disjoint 5-fold splits with 3 seeds, strata of the relative growth rate, a gradient-boosting baseline, base rates and controls, followed by external cell lines and a patient proof-of-concept on TCGA colon and rectal adenocarcinoma expression (UCSC Xena) with OptGP sampling of the adjacent-normal reference.

The scripts are the ones that produced the reported results. They are copied unchanged from the project tree, so they keep their original numbers, comments and file paths. In the code the two graph encodings are named `A` / `A'` (attention readout, the Mass Flow Graph, MFG in the paper) and `B` (mean-max readout, the bipartite metabolite-reaction graph, BMRG in the paper).

## Repository layout

```
module/crcgem/            Python package (model I/O, solver settings, MOMA, sampling, expression mapping, figure style)
scripts/00_input_data/    DepMap and TCGA expression inputs, gene universe, currency metabolites, KEGG map assignment
scripts/01_medium/        medium definitions, exchange checks, uptake caps, biomass requirement, exchange bounds
scripts/02_context_models/ tINIT extraction, essential-task check and repair, per-cell-line requirement caps
scripts/03_knockout_simulation/ pFBA reference + quadratic MOMA screens, tolerance re-solve, common-medium and lipid-uptake variants
scripts/04_dataset/       knockout sample set (graph inputs, labels, growth mask inputs)
scripts/05_training/      GNN, controls and MLP baseline training, protocol locks and lock checker, grid runners
scripts/06_baselines/     histogram gradient boosting and linear growth probe
scripts/07_evaluation/    metrics, strata, paired comparisons, controls, common-medium and lipid-uptake analyses
scripts/08_external_validation/ prediction and evaluation on the external cell lines
scripts/09_patient_poc/   patient proof-of-concept (OptGP sampling, gated MOMA, fluxome distance, GNN ensemble)
scripts/10_tables/        data and context-model statistics tables
scripts/99_checks/        installation and solver checks
scripts/check_gurobi.py   Gurobi import, version and license check
scripts/make_project_layout.sh  links the scripts into the flat layout they expect
protocol/                 locked protocol files (growth mask, protocol locks, excluded knockouts) at their original result/ paths
config/                   cell line and run templates
envs/                     conda environment exports
tests/test_core.py        unit tests of the module
```

## Installation

Three conda environments were used. They were exported with `conda env export --no-builds`.

```
conda env create -f envs/environment_crcgem.yml             # simulation: COBRApy 0.29.1, gurobipy 12.0.3
conda env create -f envs/environment_crcgem_extraction.yml  # tINIT extraction: COBRApy 0.22.1, TROPPO 0.0.7, gurobipy 11.0.3
conda env create -f envs/environment_crcgem_gnn.yml         # GNN and baselines: PyTorch 2.4.1 (CUDA 12.1), PyTorch Geometric 2.6.1
```

The exports contain pip packages that conda installs in a second step. Some packages may need adjustment on other platforms (the exports were made on Linux x86_64).

All linear and quadratic programs use Gurobi. The size-limited license bundled with the pip/conda `gurobipy` package cannot solve the context-specific models (up to about 13,000 reactions), so a full license is required. Academic users can obtain a free academic license at https://www.gurobi.com/academia/academic-program-and-licenses/. Check the installation in each solver environment:

```
conda run -n crcgem python scripts/check_gurobi.py
conda run -n crcgem_extraction python scripts/check_gurobi.py
```

The script prints the gurobipy version, solves a small LP and a small QP, and reports whether the license is full or size-limited. The `crcgem_gnn` environment does not need Gurobi.

## Paths

The scripts expect a project root with the folders `script_python/`, `script_bash/`, `module/`, `config/`, `data/` and `result/`, and they call each other by paths in that layout. Create it with

```
bash scripts/make_project_layout.sh /path/to/project
```

which links (does not copy or change) every script into `script_python/` or `script_bash/`, links `module/` and `config/`, and copies the files of `protocol/` to their `result/` paths.

The project root and the conda interpreters are hard-coded at the top of each script and were not changed for this release. Before running, replace them in the linked files:

- Python scripts: the line `ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"`, `ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")` or `R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"`. A few scripts also use the root inside other string literals and `sys.path.insert(...)` lines.
- Bash runners: `R=/data/workspace/cdgu/Winery/seeding/CRC_GEM` and the interpreter variables `PY`, `PYC`, `P` (`~/anaconda3/envs/crcgem/bin/python`), `PYG` (`~/anaconda3/envs/crcgem_gnn/bin/python`) and the `crcgem_extraction` interpreter.
- `tests/test_core.py` and `scripts/99_checks/99_verify_install.py` read an E. coli core SBML file from a fixed path (`ECOLI`). The model-based tests in `test_core.py` fail when that file is absent, and `99_verify_install.py` skips its model-based checks.
- `scripts/01_medium/30_ko_screen_medium.py` adds `bin/pylibs_highs1.15.1` (HiGHS 1.15.1 Python bindings, not included) to `sys.path`.

For example, `grep -rl /data/workspace/cdgu/Winery/seeding/CRC_GEM scripts module tests` lists every file that carries the root, and `grep -rl anaconda3 scripts` lists the files that carry an interpreter path.

## Input data

No data are included in this repository. Place the downloaded files at the paths below, relative to the project root.

DepMap Public 24Q4 (https://depmap.org/portal/data_page/, release "DepMap Public 24Q4"):

```
data/depmap/CRISPRGeneDependency.csv
data/depmap/CRISPRGeneEffect.csv
data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv
data/depmap/Model.csv
data/medium/sources/depmap_24Q4/ModelCondition.csv
```

Human-GEM v2.0.0 (https://github.com/SysBioChalmers/Human-GEM, tag v2.0.0):

```
data/scaffold/Human-GEM.xml                    the SBML model
data/scaffold/humangem_genes.tsv               gene annotation table of the model
data/scaffold/metabolicTasks_Essential.txt     the 57 essential metabolic tasks distributed with Human-GEM
```

TCGA colon and rectal adenocarcinoma (patient proof-of-concept), UCSC Xena GDC hub, STAR TPM, log2(TPM+1):

```
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.star_tpm.tsv.gz
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-READ.star_tpm.tsv.gz
https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap
```

placed in `data/tcga_xena_20261103/` (the folder is passed to `76c_tcga_expression.py --xena`, and the probe map path is its `--probemap` default).

KEGG pathway maps are fetched from the KEGG REST API (https://rest.kegg.jp) by `145_kegg_reaction_map.py` and saved under `data/kegg_<date>/`. KEGG data are subject to the KEGG terms of use.

Medium definitions. `31_build_medium_definitions.py` assembles the media from formulation sources under `data/medium/sources/` (vendor formulation pages, DepMap `ModelCondition.csv`, GDSC and KCLB culture annotations), and `52_jain_uptake_caps.py` reads the CORE uptake data of Jain et al. 2012 [ref: doi: 10.1126/science.1218595] from `data/medium/sources/jain2012/`. The collected source files are not part of this repository.

## Pipeline

Commands are run from the project root after `make_project_layout.sh` and the path edits above. Section names refer to the Methods of the paper.

1. Cell lines, labels and expression (Methods, "Cell lines, labels, context-specific models and media"). Environment `crcgem`.
   `python script_python/11_depmap_crc_trainset.py`, then `50_struct_universe.py` and `43_currency_metabolites.py`. External cell lines: `76_oob_expression.py` and `76b_external_expression.py --lines ACH-... --out data/depmap/<dir>`.
2. Media and uptake caps. `31_build_medium_definitions.py`, `45_universal_exchange_checks.py` (runner `bash script_bash/46_resume_universal_checks.sh`), `47_minimal_rescue_sets.py`, `52_jain_uptake_caps.py`, `53_biomass_requirement.py`, `57_build_uptake_bounds.py`. Scripts 47 and 53 import functions from `30_ko_screen_medium.py`.
3. Context-specific models. `54_task_protection.py`; tINIT extraction with `22_tinit_localT2.py` in the `crcgem_extraction` environment (runner `bash script_bash/55_run_tinit_all.sh <N parallel>`); essential-task repair `56_task_check_repair.py <ModelID>`; per-cell-line requirement caps `62_line_requirement_caps.py [n_proc]`.
4. Reference flux and knockout simulation (Methods, "Reference flux, knockout simulation and sample set"). `58_moma_line.py <ModelID>` per cell line (runners `59_run_repair_moma.sh`, `63_rerun_moma_linecaps.sh`), QC `61_qc_moma_outputs.py`. Tolerance re-solve `104_moma_tolpatch_line.py <ModelID>` (runner `bash script_bash/105_run_moma_tolpatch.sh crc|external|all [N]`), summary `104b_summarize_tolpatch.py`, diagnosis of non-optimal knockouts `104c_diag_nonoptimal_tolpatch.py`.
5. Sample set. `64d_gnn_build_dataset_tolpatch.py` (the derived chain is `bash script_bash/109_rebuild_derived_tolpatch.sh`, which also runs 104b, 106 and the baselines). The growth mask and the excluded non-optimal knockouts are in `protocol/result/gnn_20261028/growth_mask_v2_20261029.json` and `protocol/result/gnn_20261030_tolpatch/exclude_nonoptimal_20261030.tsv`.
6. GNN training (Methods, "Two graph encodings, the GNN and the fixed-epoch training protocol"). Environment `crcgem_gnn`, GPU. `81b_gnn_train_v2.py` is the training script; the grid runner is `bash script_bash/107_run_protocol_v3_tolpatch.sh dry|grid|monitor|check` (both encodings, both splits, seeds 0-2, folds 0-4, fixed 30 epochs, cosine learning-rate schedule). Each run is checked against its protocol lock by `101_check_protocol_v3.py`.
7. Baselines and controls (Methods, "Baselines and controls"). `86_flat_baselines_v2.py` and `90_flat_expression_v2.py` (gradient boosting with and without medium columns, linear growth probe), controls `90_gnn_e2e3_v2.py` and `93_mlp_flat_v2.py` (runner `bash script_bash/108_run_protocol_v3_controls_tolpatch.sh`), disabled-reaction indicator `122_run_flag_arm_tolpatch.sh` with lock `121_make_lock_flag_arm.py`.
8. Evaluation (Methods, "Evaluation"). `85_eval_gnn_v2.py <run_root> [--flat-root ...]` for AUROC, AUPRC and strata; `114_grid_comparison_table.py`, `118_compare_controls_all_seeds.py`, `120_measure_disabled_flag_coverage.py`, `124_flag_arm_multiplicity.py`, `129_gnn_vs_comparators_paired.py`; runner `123_eval_flag_arm.sh`.
9. Sensitivity analyses (Methods, "Sensitivity analysis and external validation"). Common medium: `91_run_moma_fixed_medium.sh` (uses `58b_moma_line_fixed.py`, `62_line_requirement_caps.py`, `64b_gnn_build_dataset_paths.py`), `112_common_medium_tolpatch_chain.sh`, `117_make_lock_common_medium.py`, `116_run_protocol_v3_common_medium.sh`, `119_eval_common_medium.sh`, `111_compare_common_medium.py`, `139_oof_intersection_common_vs_own.py`. Lipid uptake: `92_run_moma_serum_variant.sh` and `scripts/07_evaluation/lipid_uptake_sensitivity/`. Medium as GNN input: `134_make_lock_medium_input.py`, `154_make_lock_medium_input_B.py`, `135_run_medium_input_lowprio.sh`, `155_run_medium_input_B.sh`, `155b_run_medium_input_B_seeds12.sh`, evaluation `150_eval_medium_input.sh`, `150b_eval_medium_input_B.sh`. Stopping rule: `133_run_stopping_sensitivity.sh`, `157_run_stopping_sensitivity_B.sh`, `157b_run_stopping_sensitivity_B_seeds12.sh`, evaluation `144_eval_stopping_sensitivity.sh`, `150c_eval_stopping_B.sh`.
10. External validation. Context models and MOMA for external cell lines: `O=<result dir> EXPR=<expression.ENSG.parquet> LINEMAP=<lines.tsv> bash script_bash/77b_run_external_context_moma.sh [N parallel]` (and `77_run_oob_context_moma.sh`), tolerance re-solve `105b_run_external_tolpatch.sh`, line maps `105c_merge_external_line_maps.py <out_dir>`, sample set `64d_gnn_build_dataset_tolpatch.py`, comparator `86b_flat_baselines_external.py`. Prediction `99_gnn_predict_external.py --model-root <run folder> --dataset <dataset folder> --out <folder>` (runners `113_run_external_predict_tolpatch.sh`, `113b_run_external_predict.sh <gpu>`), evaluation `85b_eval_external.py --pred-roots <root> ...`, `136_external_gnn_vs_baserate.py`, `140_eval_normal_gcrc.py`, `140b_eval_gi_falsifiers.py`, input check `140c_check_external_inputs.py`.
11. Patient proof-of-concept (Results, "Patient proof-of-concept"). Expression `76c_tcga_expression.py --xena data/tcga_xena_20261103 ...`; KEGG map assignment `145_kegg_reaction_map.py`; gene lists `143_poc_gene_lists.py`; model construction, MOMA and GNN prediction `145_patient_poc_chain.sh` (runs 77b, 104, 64d, 99, 146 and the OptGP sampler `110_optgp_timing.py`); float64 normal-side samples `160_poc_normal_float64.sh` with check `161_check_optgp_float64.py`; gated MOMA re-solve `171_moma_gated.py` with check `172_check_moma_gated.py` (runners `171_poc_moma_gated_dry.sh`, `172_poc_moma_gated_full.sh`, `173_poc_gated_post.sh`, tumor side `175_poc_tumor_optgp_gated.sh`); fluxome distance `141_fluxome_distance.py` with check `148_check_fluxome_distance.py` and percentiles `174_percentile_gated.py`; re-solved knockout merge `177_merge_resolved_knockout.py` and `178_poc_gnn_with_fasn.sh`; summaries `191_poc_reference_comparison.py`, `198_poc_distance_magnitude.py`, `199_poc_per_gene_view.py`, `202_poc_gnn_targets.py`, `203_poc_target_summary.py`, `205_poc_central_pathways.py`, gene-drug table `176_poc_ko_gene_drug_table.py`. `142_moma_from_samples.py`, `158_layout_142_for_141.py` and `159_repro_error_by_objective.py` are the earlier per-sample MOMA path that `160_poc_normal_float64.sh` still calls. The reported distances come from the gated re-solve (171, 173).
12. Tables. `110_data_statistics.py --dataset result/gnn_20261030_tolpatch/dataset_v2`, `112_context_model_statistics.py`, `69_gene_counts.py`.

The usage of every script is given in its docstring or header comment.

## Outputs

Each step writes under `result/` (for example `result/moma_20261030_tolpatch/<ModelID>/` for the re-solved knockout screens, `result/gnn_20261030_tolpatch/dataset_v2/` for the sample set, `result/gnn_20261030_tolpatch/runs_v2/<run>/fold<k>/pred_test.tsv` for test predictions, and evaluation folders with `summary.tsv`, `per_fold.tsv` and `per_fold_seed.tsv`). Output tables start with `#` comment lines that state the inputs and parameters.

## Hardware

Training ran on NVIDIA GeForce RTX 3090 GPUs (24 GB). The grid runner places one BMRG run (about 15 GB) and one MFG run (about 3 GB) on one 24 GB GPU (header of `107_run_protocol_v3_tolpatch.sh`). Knockout simulation and the baselines run on CPU.

## License

MIT License, see `LICENSE`.

## Citation

[TODO - citation of the paper after acceptance]
