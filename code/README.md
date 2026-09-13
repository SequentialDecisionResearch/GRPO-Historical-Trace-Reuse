> **Post-SVAMP repository note (2026-09-13):** This README originated from the pre-SVAMP cleanup audit and is retained for provenance. The public repository now includes the completed Program 08 SVAMP external stress test, updated Program 09 paper-output layer, and `posthoc_alpha_mechanism_analysis_v2.py`. For current repository status, start with the root `README.md` and `README_REPRODUCIBILITY.md`.

# GRPO–OPE final cleaned research-code package

This archive is a **curated research/reproducibility package**, not a dump of every debugging script created during the experiment.
It is aligned to the GSM8K results produced by Program 09 on 2026-09-11.

## Current paper status

- Main GSM8K evidence chain completed: Programs 03–07 and Program 09.
- Program 09 paper-output source SHA-256: `fac8c8848a1087a5297217261f25ff2ba3fb036eab6dfd6bcdfc2dc7c48bb6bd`.
- Frozen gate SHA-256: `37c810f9c2c05b617e69dfe0b3b43561b791c1596ebce2c0fbc0f3566357f136`.
- Program 08 / SVAMP was **not included** in the current paper-output manifest. `run_program08_windows_wrapper.py` is retained only because SVAMP is the predeclared optional external-robustness extension.

## Authoritative protocol

`configs/protocol.yaml` was restored from the exact protocol file used by the study.
SHA-256: `2a4934d6036e5bc87e629dd2883dc7c38d65e0b98ad6bc248bec077ae654ecc8`.
This matches the Program 09 paper-output manifest.

Important: the YAML still contains the originally frozen `online_reference.L_audit: 32`. The official TEST computation used the separately documented post-start Program-05 sample-size amendment with effective `L_main=8, L_audit=8`. Do **not** silently rewrite the frozen protocol YAML to 8/8; the amendment is part of the provenance chain.

## Files retained

### Core research programs
`00_download_and_manifest.py` through `09_make_paper_outputs.py`, plus `freeze_protocol.py` and `check_environment.py`.

### Scientific amendment / design-lock evidence
- `amend_program04_preanalysis.py`
- `benchmark_test_sample_size.py`
- `benchmark_pair_grid_reduction.py`
- `prepare_reduced_grid_amendment.py`
- `benchmark_program05_reference_sample_size.py`
- `prepare_program05_sample_size_amendment.py`
- `prepare_program06_rolling_comparator_repair.py`

These are retained because the final official-test computation did not simply use the original 117-pair / original-reference design. The paper must disclose those audited amendments rather than hide them.

### Final execution helpers actually used by the final numerical path
- Program 04 reduced-grid + fresh-process execution and the deterministic 9-comparator structural repair.
- Program 05 final 8/8, three-process execution (`*_v2` / `*_fixed` versions).
- Program 06 repaired 45-pair wrapper.
- Windows runtime wrappers for Programs 08/09.

`run_program04_rolling_repair_chunk.py` is intentionally retained: although the earlier rolling-repair batch supervisor was superseded, the final direct-repair supervisor calls this Python file for the final full 45-pair index rebuild and verify-only pass.

### Numerical/identity audit tools retained
The four Program-04 identity/FP32 diagnostic scripts are retained as audit evidence for the pre-analysis scoring amendment. They are not part of the normal paper-output run.

## Windows wrappers

The frozen Program 09 source must remain byte-identical to the source recorded by the paper-output manifest. On Windows, its figure durability flush can fail when `os.fsync()` is called on a read-only descriptor. `run_program09_windows_wrapper.py` applies only an in-memory compatibility patch and leaves the frozen source file unchanged.

Program 08 has an analogous Parquet flush issue on Windows. `run_program08_windows_wrapper.py` applies the same kind of runtime-only compatibility fix. Program 08 is optional and its results are not in the current paper bundle.

## Important missing artifacts — collect before a public reproducibility release

The four uploaded ZIPs are enough to write the current paper, but **not enough for a full from-scratch public reproduction**. At minimum, copy the following from the workstation before final GitHub/archival release:

- `manifests/protocol_lock.json` (expected SHA-256 `762c1727cee6f4fec6b65cbd649654376e4bb9521f782fbc52ec53ddb3fd9dfa`)
- `outputs/frozen_gate.json` (expected SHA-256 `37c810f9c2c05b617e69dfe0b3b43561b791c1596ebce2c0fbc0f3566357f136`)
- `manifests/frozen_gate_manifest.json` (expected SHA-256 `1e52aac5325878b3eee2bc57925c11d715507c211bc883454c664c61f206450d`)
- `outputs/diagnostics/program06_test_manifest.json` (expected SHA-256 `13550c8e34f681be0ee109818c5d95307a12cf364678845169a7c4f058a1a84a`)
- `outputs/diagnostics/program07_validate_manifest.json` (expected SHA-256 `c0d4c384e9191e465b021f42e8268257183158aa1da059ea53d65dd3a530f2cd`)
- `outputs/diagnostics/program07_test-only_manifest.json` (expected SHA-256 `934baed5229ffe78fa273c2fc5c150df87a75eed8ce3d2649186f55608d3860d`)
- Program 00/01 provenance: `data_manifest.json`, `model_manifest.json`, `environment_manifest.json`, `split_registry_manifest.json`
- all final amendment/selection manifests referenced by the retained wrappers (Program 04 reduced grid, Program 05 8/8, Program 06 comparator repair)
- `outputs/tables/T03_overlap.csv` (198,192 rows; expected SHA-256 `5a933c4b3c2cfacd36bd4c5b3a59b6891064996ece7532a9c43b91d589212348`)
- the 63 Program-05 development `reference_summary.json` records listed in `EXPECTED_PROVENANCE_FROM_PAPER_MANIFEST.json`
- `tests/test_correctness_firewall.py` if you still have the original file. The old delivery audit claimed 10 tests passed, but the test source itself was not present in the uploaded code ZIP, so this cleaned archive does not pretend that it is included.

Large raw datasets, model checkpoints, behavior logs, target-rescore shards and online-reference Parquet shards should normally **not** be committed directly to GitHub. Publish them separately (release/archival storage) if full bit-for-bit reproduction is desired, and keep their hashes/manifests in the repository.

## Do not use as final evidence

See `REMOVED_OR_SUPERSEDED_FILES.md`. Older failed/benchmark-only runners were removed so a future reader does not accidentally rerun a known-obsolete branch.
