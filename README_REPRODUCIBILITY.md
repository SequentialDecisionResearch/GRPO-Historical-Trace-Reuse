# Reproducibility Guide

## 1. What this repository is designed to reproduce

The repository is designed to make the paper-facing evidence chain inspectable and to preserve the frozen scientific outputs used in the manuscript:

- frozen GSM8K protocol and amendment manifests;
- official-test diagnostics and frozen gate outputs;
- frozen SVAMP external results (`outputs/svamp/`);
- deterministic Program 09 paper outputs (`outputs/paper/`);
- post-hoc alpha mechanism outputs (`outputs/posthoc_alpha/`);
- exact paper source and publication figures (`paper/source/`).

## 2. Paper build

```bash
cd paper/source
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The build should produce `main.pdf` matching the public paper content and pagination.

## 3. Code environment

The curated code package is under `code/`.

Primary environment files:

- `code/grpo_ope_environment.yml`
- `code/requirements.txt`
- `configs/protocol.yaml`

The numbered research programs run from `00_download_and_manifest.py` through `09_make_paper_outputs.py`. Several amendment and Windows-runtime wrappers are retained because they belong to the actual provenance chain. Read `code/POST_SVAMP_UPDATE.md`, `code/REMOVED_OR_SUPERSEDED_FILES.md`, and the manifests before rerunning any historical branch.

## 4. Frozen external layer

The final SVAMP results are under:

```text
outputs/svamp/
```

The final deterministic paper layer is under:

```text
outputs/paper/
```

The exploratory mechanism layer is under:

```text
outputs/posthoc_alpha/
```

Do not refit the gate or reselect the frozen SVAMP pair registry when reproducing the reported external claims.

## 5. Intentionally omitted large artifacts

This GitHub-ready package intentionally omits large artifacts that are unsuitable for ordinary Git history, including model checkpoints and large raw/intermediate Parquet/CSV collections.

`T03_overlap.csv` is intentionally omitted. The curated pre-SVAMP code audit records its expected SHA-256 as:

```text
5a933c4b3c2cfacd36bd4c5b3a59b6891064996ece7532a9c43b91d589212348
```

For a full archival reproduction, store large artifacts in Zenodo, an institutional repository, or release assets and preserve their hashes in this repository.

## 6. Chronological evidence discipline

The interpretation of the repository depends on the order in which evidence was frozen:

1. development analyses and diagnostic calibration;
2. held-out seed evaluation;
3. untouched GSM8K official-test evaluation;
4. frozen SVAMP external prompt-distribution stress test;
5. explicitly post-hoc exploratory alpha mechanism analysis.

The alpha analysis must not be described as a pre-registered confirmatory test.

## 7. Integrity

`REPO_SHA256SUMS.txt` contains SHA-256 hashes for the public package. Recompute it before every tagged release after intentional changes.
