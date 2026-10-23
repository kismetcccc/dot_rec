# LGMRec data and metric alignment

Static verification date: 2026-10-23. Reference snapshot: [d66655fc4a2b066f915365b58e806f39cd08f0dd](https://github.com/georgeguo-cn/LGMRec/tree/d66655fc4a2b066f915365b58e806f39cd08f0dd). This is an independent implementation of metric mathematics and explicit data interfaces, not copied upstream source or a bitwise reproduction. No actual features were downloaded; no training or benchmark was run.

## Data contract and evidence

Use the original baby.inter, sports.inter and clothing.inter, with TAB-separated userID/itemID/x_label and existing labels 0=TRAIN, 1=validation, 2=test. Do not replace these splits by a fresh 8:1:1 random split merely because the paper describes that ratio. The [dataset implementation](https://github.com/georgeguo-cn/LGMRec/blob/d66655fc4a2b066f915365b58e806f39cd08f0dd/utils/dataset.py) selects labels directly.

Three templates are provided under configs/dataset/{baby,sports,clothing}.manifest.example.json. They contain historical-log catalog declarations, NOT locally verified datasets. Every actual file SHA256 is deliberately a placeholder, rejected before data-file access. Reference Git blob SHA1 identifies a repository object and is never accepted as a file SHA256. Confirm counts, zero-based ID mapping, feature row alignment, natural missingness and hashes before use. Amazon release, exact preprocessing and exact text encoder version remain unknown where not established.

| Dataset | Log users/items | Log TRAIN/valid/test rows | Reference .inter Git blob SHA1 (not SHA256) |
|---|---|---|---|
| Baby | 19445 / 7050 | 118551 / 20559 / 21682 | a2bfc59e9a5f25579b2d27e55e29713853a46ca0 |
| Sports | 35598 / 18357 | 218409 / 37899 / 40029 | f01d7e95dbd8e7b91859633127e6b2db3af0e8ad |
| Clothing | 39387 / 23033 | 197338 / 40150 / 41189 | aec537ffb7e41decc2bf8f2d48e8b4931aca740e |

Sources: [three dataset configurations](https://github.com/georgeguo-cn/LGMRec/tree/d66655fc4a2b066f915365b58e806f39cd08f0dd/configs/dataset) and [historical logs](https://github.com/georgeguo-cn/LGMRec/tree/d66655fc4a2b066f915365b58e806f39cd08f0dd/log). These are log counts, not counts from downloaded interaction files in this work. The repository README directs feature acquisition to Google Drive and identifies VGG-16 image features and Sentence-Transformers text features. Configured filenames are image_feat.npy and text_feat.npy; logs show dimensions 4096 and 384 respectively. Actual feature bytes, completeness and dimensions have not been verified. No template is ready to run unchanged.

## Preset and exact metric mathematics

Use configs/model/bast_lgmrec.json: training.metrics=[recall,ndcg,precision,map], training.cutoffs=[5,10,20,50], selection_metric=recall@20, test_mask_valid=false, evaluation_population=train_users_only. Other BAST settings remain unvalidated debugging defaults, not transferred LGMRec hyperparameters. The existing bast.json defaults remain Recall/NDCG@10/20 with population all.

For one user with P positive targets and rank hit indicators h_r, H_r=sum_{s<=r} h_s:

- Recall@K = H_K/P.
- Precision@K = H_K/K, even if fewer than K eligible candidates remain.
- NDCG@K = sum_{r<=K} h_r/log2(r+1) divided by sum_{r<=min(K,P)} 1/log2(r+1).
- AP@K = sum_{r<=K} h_r*(H_r/r) / min(K,P); MAP@K is the mean of AP@K over users.

Each metric is user-macro, not micro. Users with no target positives are outside the evaluation population. Missing returned positions and -1 padding are misses; duplicate valid ranked IDs are rejected. [Upstream metrics](https://github.com/georgeguo-cn/LGMRec/blob/d66655fc4a2b066f915365b58e806f39cd08f0dd/utils/metrics.py) and [overall settings](https://github.com/georgeguo-cn/LGMRec/blob/d66655fc4a2b066f915365b58e806f39cd08f0dd/configs/overall.yaml) establish this convention. Resolved configuration stores metrics/cutoffs and both validation and explicit final testing use them. Duplicate/unknown metric names, duplicate/nonpositive cutoffs and selection metrics absent from the configured report are errors.

The main paper comparison reports only Recall@10/20 and NDCG@10/20. The repository-complete report contains all 16 combinations. Preserve full precision in JSON; use four decimals only when presenting a table. No display-rounding CLI was added.

## Population and deliberate differences

train_users_only filters evaluation target rows whose user has zero TRAIN degree, retaining cold items and all catalog candidates. Validation and test share this rule. It never edits source files, hashes, TRAIN support or masks. Coverage retains raw target counts and adds raw/effective/filtered distinct-user and interaction counts plus effective cold-item counts. Empty effective populations are errors. Existing all and warm_only semantics are unchanged: all retains cold users, while warm_only rejects cold endpoints without filtering.

Upstream filters cold users during split loading; this implementation filters explicitly during evaluation. The resulting target population is equivalent for the same valid inputs, not bitwise reproduction. Provenance reports both raw and effective populations. The final evaluator applies the same saved configuration. Histories masked at validation and test contain TRAIN only, matching upstream quick_start lines 37-38; test_mask_valid=false is explicit. We retain negative infinity instead of upstream -1e10, and deterministic lower-item-ID tie breaking instead of unspecified torch.topk ties. Full-catalog candidates remain unchanged. Our negative sampler uses the declared catalog whereas upstream samples TRAIN-seen items; this is a disclosed training difference, not a claim of complete experimental replication.

Upstream quick_start lines 81-83 select the outer BEST by test performance. We do not copy that process: main.py is validation-only, and evaluate.py tests one explicitly selected run after selection freezes, with source/data/checkpoint bindings and overwrite refusal. No upstream stochastic Gumbel evaluation procedure is adopted. Comparisons require the same actual data and protocol, not merely matching dataset names.

## External paper reference only: NOT BAST results

[Paper v2 Table 2](https://arxiv.org/html/2312.16400v2#S4.T2) reports the following LGMRec numbers. They are external published reference values, not reproduced measurements, targets to chase, or evidence that our runs are directly comparable.

| Dataset | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| Baby | .0644 | .1002 | .0349 | .0440 |
| Sports | .0720 | .1068 | .0390 | .0480 |
| Clothing | .0555 | .0828 | .0302 | .0371 |

Repository BEST summaries and single-epoch logs may differ from the paper. Keep those three source categories separate; do not splice favorable values into a shared baseline row. BAST results: **NOT RUN** for every dataset and metric.

## External workflow only

After verifying local files and replacing template declarations/hashes, use main.py with --config configs/model/bast_lgmrec.json, --manifest <verified-manifest>, --output <new-run>, --execute-training. Select using validation only; then separately invoke evaluate.py --run <selected-run> --manifest <verified-manifest> --execute-final-test. Neither action is authorized on the development computer. No automatic feature downloads, split regeneration, hyperparameter search or training CI is included.
