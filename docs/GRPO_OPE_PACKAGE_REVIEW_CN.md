# GRPO–OPE 四个 ZIP 整理复核报告

## 结论

你上传的四类材料已经足够**马上开始写文章**，其中 `paper.zip` 是当前最重要的数值/图形权威来源；但它们还不是“公开 GitHub 后别人可以从零完整重现”的完整档案。最重要的缺口是 provenance manifests、T03、development online-reference summaries，以及 editable manuscript LaTeX source。

## 1. 代码 ZIP

原 ZIP 共 63 个文件。清理后保留 37 个 Python/BAT 执行/审计文件，加上准确 protocol、环境文件和新的审计说明。

核心 00–09 Python 文件全部保留且未改动；所有保留 Python 文件均通过语法编译检查。删掉了明确失败、已被 supersede、纯速度 benchmark 或与本文无关的文件。特别是：

- 删除 `run_program06_reduced_8x8.py`：它是已知结构不完整的 36-pair P06 路径；最终结果来自 repaired 45-pair path。
- 删除 `patch_program09_windows_fsync.py`：它会修改 frozen Program09 source，已知会触发 source-hash 保护；保留 runtime wrapper。
- 删除旧 P05 workers/finalizers/supervisors，只保留最终 8/8 + parallel3 fixed + v2 路径。
- Program04 的旧 rolling repair BAT 删除，但 `run_program04_rolling_repair_chunk.py` 保留，因为最终 direct supervisor 仍调用它做全 45-pair finalization/verify。
- 修正 `run_program04_chunked.bat` 一处纯批处理语法错误：`set CHUNK=20"` → `set "CHUNK=20"`；科学参数仍是 20，没有改变任何数值定义。

此外我从你的 Library 恢复了**与当前最终结果完全匹配的 `configs/protocol.yaml`**，SHA-256 为 `2a4934d6036e5bc87e629dd2883dc7c38d65e0b98ad6bc248bec077ae654ecc8`，与 Program09 manifest 一致。

## 2. tables.zip

T01/T02/T04/T05/T06 五张 CSV 与 Program09 manifest 的行数、字节数和 SHA-256 **全部完全一致**。因此它们可以安全作为正文核对/附录证据。

但原 `tables.zip` 缺 `T03_overlap.csv`。Program09 明确记录 T03 为 198,192 行、70,086,283 bytes、SHA-256 `5a933c4b3c2cfacd36bd4c5b3a59b6891064996ece7532a9c43b91d589212348`。所以当前 tables bundle 是 compact verified bundle，不是完整再生成 bundle。

## 3. paper.zip

这是目前最干净、最权威的一包。`paper_output_manifest.json` 列出的 29 个生成文件全部逐一校验，**0 个 hash/size mismatch**。我没有修改 `paper/` 里面任何生成文件，只在 ZIP 外层增加 verification README 和 checksum。

当前 manifest 明确记录：official GSM8K test 已进入主输出；SVAMP=False；second-model=False。以后 Program08 若完成，不要手改这些文件，而是重跑 Program09，另存一个新的 versioned output ZIP。

## 4. paper_placeholder.zip

保留新版 `GRPO_Historical_Trace_Reuse_Frozen_Placeholder_Manuscript_v2.pdf`；删除旧 placeholder PDF。保留理论分配备忘录。`gpolamma.pdf` 放到 `theory_reference/`，因为其 DR/cross-fitting 理论主体属于另一篇 contextual-bandit paper，只能作为思想参考，不能直接冒充本文定理。

最重要的缺口：placeholder ZIP 没有对应 LaTeX source tree。写文章时 PDF 可以作为结构参考，但 GitHub/正式持续修改最好加入原始 `main.tex + sections + appendices + bibliography`。

## GitHub / 可复现实验还需要补什么

建议 GitHub 放：最终清理代码、准确 config、小型 manifests、T01–T06（含 T03；若 70MB 超过普通 GitHub 单文件舒适范围可用 Git LFS/Release）、Program09 paper outputs、论文 LaTeX source、README。

不要直接把 checkpoints、raw model、全部 behavior/rescore/online Parquet shards 塞进 GitHub。若要 bit-for-bit 全复现，把大资产放 Release/Zenodo/OSF/Hugging Face Dataset 等外部归档，并在 GitHub manifests 中记录 URL + SHA-256。

当前最应从 `C:\lsg\grpo_ope_reuse` 再备份出来的是：protocol lock、frozen gate/manifest、Program06/07 manifests、Program04/05/06 amendment-selection manifests、T03、63 个 development reference summaries，以及 correctness-firewall test source。精确已知 hash 已写进 cleaned code ZIP 的 README / `EXPECTED_PROVENANCE_FROM_PAPER_MANIFEST.json`。
