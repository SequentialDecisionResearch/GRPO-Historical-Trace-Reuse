# Public Release Checklist

## Repository preparation

- [x] Publication-grade directory structure created.
- [x] Final public paper included.
- [x] Compilable LaTeX source included.
- [x] Post-SVAMP code included.
- [x] Frozen SVAMP outputs included.
- [x] Program 09 paper outputs included.
- [x] Post-hoc alpha mechanism outputs included.
- [x] Protocol/manifests included.
- [x] Large-file policy documented.
- [x] `T03_overlap.csv` excluded from GitHub-sized package.
- [x] Dual licensing documented.
- [x] `CITATION.cff` added.

## Before making GitHub public

- [ ] Create GitHub repository `GRPO-Historical-Trace-Reuse`.
- [ ] Upload this repository tree, not the outer ZIP as a single file.
- [ ] Confirm no secrets/tokens/local machine credentials are present.
- [ ] Review author name/email and decide whether both should be public.
- [ ] Confirm the public paper is the intended final version.
- [ ] Recompute `REPO_SHA256SUMS.txt` after any edits.
- [ ] Commit with a clear message, e.g. `Public paper and reproducibility package v1.0`.
- [ ] Create tag `v1.0-paper`.
- [ ] Create a GitHub Release from `v1.0-paper`.

## Archival DOI

- [ ] Connect the GitHub repository to Zenodo.
- [ ] Enable archiving for the repository.
- [ ] Publish the GitHub `v1.0-paper` release.
- [ ] Record the minted DOI in `CITATION.cff`, README, and public paper source.
- [ ] Create a small follow-up release only if DOI metadata requires correction; do not silently rewrite the scientific evidence layer.

## Communication layer

- [ ] Prepare a Medium explainer after the GitHub release is stable.
- [ ] Link Medium -> GitHub/public paper.
- [ ] Add Medium URL to the public paper only if desired.
- [ ] Keep the scientific claims aligned with the frozen manuscript.

## Peer review / TMLR

- [ ] Send public PDF + GitHub repo for informal expert review.
- [ ] Track substantive reviewer feedback separately from publicity edits.
- [ ] Prepare a separate anonymous TMLR manuscript.
- [ ] Prepare anonymous supplementary material with names, email, Medium URL, and identifiable GitHub URL removed.
