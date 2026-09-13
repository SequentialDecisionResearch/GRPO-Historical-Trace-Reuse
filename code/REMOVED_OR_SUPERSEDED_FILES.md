# Removed or superseded files

The following files were present in the uploaded code ZIP but intentionally excluded from the cleaned archive.

- `09_make_paper_outputs.pre_windows_fsync_patch.py` — Byte-for-byte duplicate of the restored final `09_make_paper_outputs.py`; redundant.
- `DELIVERY_AUDIT.txt` — Original audit was stale/incomplete relative to the final runs; replaced by `FINAL_PACKAGE_AUDIT.md`.
- `README.md` — Original README was incomplete for the final amended execution path and referenced files absent from the uploaded ZIP; replaced by the new README.
- `SOURCE_SHA256.txt` — Contained stale source hashes for several core files and hashes for files not present in the uploaded ZIP; regenerated as `FINAL_SOURCE_SHA256.txt`.
- `benchmark_program04_pipeline.py` — Runtime benchmark only; not a scientific design decision in the final paper.
- `benchmark_program04_scoring.py` — Runtime benchmark only; not a scientific design decision in the final paper.
- `china_ai_patent_template.pdf` — Unrelated to the GRPO–OPE paper.
- `codeguide.pdf` — Not required by the final numerical path or retained scientific-audit path.
- `diagnose_program04_sequence_lengths.py` — Runtime diagnosis only; not needed for final numerical reproduction.
- `finalize_program05_8x8.py` — Superseded by `finalize_program05_8x8_v2.py`.
- `finalize_program05_reduced_grid.py` — Pre-8/8 finalizer; superseded by the final 8/8 design.
- `patch_program09_windows_fsync.py` — Known wrong final approach: it modifies the frozen Program-09 source and therefore triggers the protocol/source-hash protection. The runtime wrapper is the correct Windows workaround.
- `run_program04_rolling_repair_chunked.bat` — Superseded slow repair supervisor. The Python chunk file is still retained because the final direct supervisor uses it for finalization/verification.
- `run_program05_8x8_chunk_worker.py` — Superseded by `run_program05_8x8_chunk_worker_v2.py`.
- `run_program05_8x8_chunked.bat` — Superseded by the final three-process fixed supervisor.
- `run_program05_8x8_parallel3.bat` — Superseded by the fixed version.
- `run_program05_8x8_parallel3.py` — Superseded by `run_program05_8x8_parallel3_fixed.py`.
- `run_program05_chunk_worker.py` — Pre-8/8 worker; superseded by final 8/8 worker.
- `run_program05_chunked.bat` — Pre-8/8 supervisor; superseded.
- `run_program05_specific_chunk.py` — Throughput-test helper, not part of the final numerical/provenance path.
- `run_program06_reduced_8x8.py` — Known structurally incomplete 36-pair Program-06 attempt; it produced no rolling-comparison rows for retained rolling targets because required 0→target comparators were absent. Superseded by the 45-pair repaired runner.
- `test_program05_8x8_parallel3.py` — Performance benchmark only; not needed to reproduce paper values.
- `test_program05_parallel2.bat` — Performance benchmark only; not needed to reproduce paper values.
- `test_program05_parallel2_fixed.bat` — Performance benchmark only; not needed to reproduce paper values.
