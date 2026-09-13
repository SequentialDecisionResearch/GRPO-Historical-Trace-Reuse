# When Can Historical GRPO Reasoning Traces Be Reused?

**Off-Policy Evaluation and Trace Refresh under Policy Drift**

Shenggang Li — Independent Researcher

This repository accompanies the paper **“When Can Historical GRPO Reasoning Traces Be Reused? Off-Policy Evaluation and Trace Refresh under Policy Drift.”** It contains the paper source, the curated GRPO/OPE research code, frozen protocol and provenance manifests, paper-facing outputs, the frozen SVAMP external stress test, and the post-hoc reward-stratum mechanism analysis.

## Research question

GRPO training creates expensive reasoning traces while the policy continues to change. Reusing old traces can reduce sampling cost, but once the policy moves, those traces become off-policy evidence. This project studies:

1. how historical sequence-level overlap deteriorates along GRPO training paths;
2. whether replacing a stale behavior log with a more recent log repairs evidence quality for the **same target checkpoint**;
3. how second-moment overlap and reward-residual allocation jointly enter the leading prompt-WIS risk; and
4. whether a scalar rESS threshold is sufficient as an operational reuse certificate.

## Main findings

- **Matched GSM8K refresh:** mean pair-level median rESS increases from **0.127 to 0.959** in **9/9** official-test comparisons; mean absolute finite-reference discrepancy falls from **0.005866 to 0.001642** in **8/9** cases.
- **Frozen SVAMP external stress test:** mean median rESS increases from **0.142 to 0.965** in **6/6** matched comparisons; mean absolute discrepancy falls from **0.00856 to 0.00230** in **5/6** cases.
- **Scalar gate limitation:** the frozen rESS threshold accepts all SVAMP cases and false-accepts **25.0%** at the 1% discrepancy tolerance.
- **Exploratory mechanism analysis:** the reward-stratum allocation factor often moves in an adverse direction, while the aggregate empirical leading-risk proxy improves in **45/45 development** and **9/9 official-test** matched comparisons. The observed pattern is therefore consistent with **overlap restoration dominating adverse reward-stratum movement**.

The paper deliberately distinguishes finite-reference **discrepancy** from theoretical error relative to the latent finite-prompt target value.

## Repository map

```text
GRPO-Historical-Trace-Reuse/
├── README.md
├── README_REPRODUCIBILITY.md
├── RELEASE_CHECKLIST.md
├── CITATION.cff
├── LICENSE
├── LICENSE-CODE
├── LICENSE-DOCUMENTATION
├── paper/
│   ├── GRPO_Historical_Trace_Reuse_final.pdf
│   └── source/
├── code/
├── configs/
├── data/
│   └── splits/
├── manifests/
├── outputs/
│   ├── diagnostics/
│   ├── paper/
│   ├── svamp/
│   └── posthoc_alpha/
├── tests/
└── docs/
```

## Compile the paper

From `paper/source/`:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The resulting `main.pdf` reproduces the current 23-page public paper layout. See `paper/source/BUILD_INSTRUCTIONS.txt` for details.

## Scientific evidence layers

The repository preserves the study’s chronological separation:

```text
development
    -> held-out training seed
    -> untouched GSM8K official test
    -> frozen SVAMP external prompt-distribution test
    -> explicitly post-hoc exploratory mechanism analysis
```

The SVAMP pair registry, estimator, scoring contract, and gate were frozen before SVAMP OPE/reference outcomes were inspected. The alpha mechanism analysis is retained as **exploratory post-hoc evidence**, not as a newly calibrated gate or confirmatory test.

## Reproducibility scope and large artifacts

This GitHub-sized repository contains the frozen scientific outputs needed to inspect the paper-level claims and provenance chain. It intentionally does **not** commit model checkpoints or the largest raw/intermediate experiment artifacts.

In particular, `T03_overlap.csv` is intentionally omitted because of its size. Its expected SHA-256 is recorded in the code audit material. Large raw behavior logs, target-rescore shards, online-reference shards, and model checkpoints should be archived separately if full bit-for-bit reproduction is desired.

See `README_REPRODUCIBILITY.md` for the exact scope.

## Versioning and provenance

The recommended first public release tag is:

```text
v1.0-paper
```

Before creating the release, verify `REPO_SHA256SUMS.txt`, the protocol/amendment manifests, and the final paper hash. A later Zenodo archive can be connected to the GitHub repository to mint a DOI for the release.

## Licensing

- **Code:** MIT License — see `LICENSE-CODE`.
- **Paper, figures, and research documentation:** CC BY 4.0 — see `LICENSE-DOCUMENTATION`.

The root `LICENSE` explains the dual-license boundary.

## Citation

See `CITATION.cff`. A DOI field can be added after the first archival release is deposited.
