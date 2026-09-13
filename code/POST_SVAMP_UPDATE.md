# Post-SVAMP code update

This directory starts from the pre-SVAMP clean code package and adds/replaces the code used by the completed SVAMP and post-hoc mechanism work.

## Updated or added scientific/runtime code

- `08_run_svamp_robustness.py` - current Program 08 source used with the FP32 repair runtime.
- `run_program08_fp32_repair.py` - preserves the failed BF16 attempt as audit evidence and writes repaired paper assets under `data/svamp_robustness_fp32_v2/`.
- `run_program08_behavior_fresh_resume.py` - resume helper for fresh behavior-asset generation during the Program 08 repair workflow.
- `09_make_paper_outputs.py` - frozen Program 09 source.
- `run_program09_windows_wrapper_v2.py` - Windows runtime wrapper used for the successful final Program 09 run; treats blank gate booleans on IS/not-applicable T08 rows as NA while preserving prompt-WIS boolean invariants, and applies the Windows fsync compatibility patch at runtime without modifying Program 09 source.
- `posthoc_alpha_mechanism_analysis_v2.py` - final low-cost exploratory reward-stratum / alpha mechanism analysis. V2 distinguishes promptwise geometric heterogeneity diagnostics from the arithmetic prompt-average aggregation required by the paper's Corollary 2 interpretation.

The older `run_program09_windows_wrapper.py` remains in the inherited package for chronology. For the final paper-output reproduction on Windows, use `run_program09_windows_wrapper_v2.py`.

## Evidence boundary

The post-hoc alpha program is exploratory and does not refit the frozen gate, select new target/behavior pairs, call the model, or mutate upstream Program 01-09 results.
