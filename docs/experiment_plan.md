# Experiment plan: external execution only

Status: all real-data experiments are **NOT RUN**. This plan targets candidate commit `9bb81bbe53d4a5b4c3da207c49163ae41caf6e98`; future changes require a new source fingerprint. Training is prohibited on the development computer. GPU training remains statically reviewed only. No experiment scheduler, automatic training CI or dataset downloader is provided.

## Freeze inputs before comparing models

Choose the actual Amazon Baby/Sports/Clothing release and record source URL, release/version, filtering, split construction, zero-based user/item mapping, encoder/version, feature dimensions, natural availability, experiment mask and SHA256 for every input file. Same dataset names do not establish matching populations or splits. Do not mix MMRec/ChaoRec variants or quote their published numbers as matched baselines without reconstructing their protocols.

Use the local manifest schema and loader. Unknown provenance must remain explicitly declared unknown; it weakens comparability rather than becoming an invented version. Reject duplicate/cross-split edges. H and negative sampling use TRAIN only; negatives do not consult test positives. Lock full-catalog candidates and whether test masks validation history (`training.test_mask_valid`, default false). Validation alone selects checkpoints and hyperparameters. Freeze the comparison before interpreting final test reports.

MTMT and FTMT are separate experiments: MTMT training visibility is natural AND experiment; FTMT training visibility is natural. Both evaluate with the identical fixed natural AND experiment mask for validation/test, on unchanged TRAIN support. FTMT cannot reveal naturally absent features. The loader checks all training-visible values for finiteness.

Use multiple predeclared model seeds, for example 2026/2027/2028, with the same fixed masks across compared variants. Record mask seeds separately; do not regenerate masks from validation/test labels. If varying mask seeds, pair each mask realization across all variants and distinguish mask variability from initialization variability. Report individual seeds and mean/spread; do not select the best test seed. Start on one fully specified dataset before expanding to the other two.

## Three prioritized stages

| Stage | Minimum comparison | Decision sought |
|---|---|---|
| 1: observed transport | Same-protocol CF, fixed-LOO and learned-LOO; M2 off, complete experiment mask initially | Does M1 improve on local content and fixed transport, or only add cost? A useful extra control is eta=0 local content. |
| 2: missing transport | 2x2 learned_group false/true x compensation false/true; experiment missing rates 0/25/50/75%, visual-only, text-only and independent patterns | Does M2 add value beyond M1 and CF under controlled missingness? Report MTMT and FTMT separately. |
| 3: falsification | q=1, no_agreement, no_support; then matched constant gate and shuffled gate/donor controls | Is the signal more than generic attenuation, coverage, or unrelated donor statistics? |

Stage 2 is a prioritized grid, not permission for unrestricted search. Begin with 0 and 50% on one dataset, then fill the declared rates and datasets if the comparison is informative and the user budgets external compute. Afterwards stress high missingness, coupled missingness, and interleaved image-only/text-only visibility. With initially complete inputs, independent dropout at rate r gives expected paired visibility (1-r)^2; coupled dropout gives 1-r; interleaving can give zero paired visibility despite 50% visibility in each modality. Report actual counts because natural missingness and degree distributions change these expectations.

Use the same small, predeclared validation tuning budget per variant. Debugging defaults are not tuned recommendations. Hold CF depth, ID dimension, dropout, regularization definition, split and masks fixed for mechanism ablations. Do not use a test result to decide another search round. The training CLI is validation-only and never evaluates test. Compare validation reports, freeze one run, then invoke the separate final-test entrypoint below. It does not discover runs or select by test score. Each run refuses a second final_test.json; cross-run selection discipline still belongs to the researcher.


## Implemented settings versus future work

| Control | Current support |
|---|---|
| CF representation baseline | Set model.alpha=0, model.beta=0, model.compensation=false. Same CF encoder/zero-degree policy; projection and gate allocation plus L2/compute remain, so this is not a lean external LightGCN timing baseline. |
| Fixed/learned LOO | model.learned_group=false/true; same allocated parameters. |
| M2 off/on | model.compensation=false/true. |
| Local-content control | model.eta=0, model.compensation=false; no transported observed residual. |
| q variants | model.quality_mode is full, unit, no_support or no_agreement. unit preserves donor/target eligibility; it does not invent donors. |
| Protocol/seeds | training.protocol=mtmt/ftmt and training.seed. |
| Fixed mask generation | Python API fixed_mask(natural, rate, seed, pattern), patterns independent/coupled/visual/text. Returns boolean availability; persist the array and its hash in the manifest. No mask-generation CLI. |
| Interleaved stress mask | Prepare and review externally; no built-in interleaved pattern switch. |
| Matched constant attenuation, shuffled gate/donor controls | Not implemented. Require separately reviewed changes and a precise permutation/matching rule; no existing CLI flag. Match attenuation using TRAIN statistics, not test outcomes. |
| Automatic subgroup/degree/abstention reports | Not implemented as a CLI wrapper. encode(diagnostics=True) exposes behavior/z/h/c/g/q, but collection, degree bins and aggregation need external reviewed tooling. |

Settings belong in JSON model/training sections; there are no command-line overrides such as --protocol, --missing-rate, --quality-mode or --seed. Unknown configuration keys raise errors.

## Metrics, diagnostics and safeguards

Current evaluator reports user-macro Recall/NDCG at 10/20 and deterministic lower-item-ID tie breaking. TRAIN histories are masked to negative infinity; optional validation-history masking at test is explicit. Full catalog remains the candidate set. Default population=all retains zero-TRAIN-degree targets and reports coverage. warm_only rejects any cold target endpoint rather than filtering silently. Cold users produce tied scores and must not be presented as solved cold-start.

For missingness subgroups, filter target positives by target item visibility (both visible, image absent only, text absent only, both absent); retain the same full candidate catalog and history exclusions. Users without positives in that subgroup are excluded from its macro average and their counts must be reported. Do not average subgroup macro scores to reconstruct the overall macro score. This subgroup wrapper is not currently implemented. Degree-group cutoffs must be defined from TRAIN only, fixed before viewing test results, and accompanied by user/item/interaction counts.

Collect paired-donor coverage, eligible target membership counts, donorless fractions, exact-zero q abstention, q distribution, ||t|| and ||c|| distributions, semantic residual norms, and degree-conditioned performance. Distinguish ineligible targets from eligible targets with rejected evidence. Complete-modality M2 should be exactly inactive; interleaved no-paired cases should give zero compensation. Do not report low variance as calibrated confidence. Sparse paired support is a design limitation, not a guarantee of improvement.

Timing/memory comparisons must include full-graph projections and propagation every minibatch and the per-tensor mean-square L2 cost. Never cache embeddings across optimizer steps. GPU determinism settings may still encounter unsupported deterministic operations; do not silently relax reproducibility or claim GPU validation from CPU unit tests.

## Exact entrypoint, checked statically

Dependencies declared in pyproject.toml: Python >=3.10, torch >=2.6,<3, numpy >=1.24. From the repository root, make the repository root importable (PowerShell: `$env:PYTHONPATH = (Get-Location).Path`). In a separately authorized external execution environment only:

```text
python main.py --config configs/model/bast.json --manifest /external/dataset/manifest.json --output /external/new-run --execute-training
```

The three path options are required and the output directory must not exist. Without --execute-training the parser refuses execution. Paths above are placeholders, not local files or runnable data supplied by this project. Device is training.device in JSON; no --device option. This documentation pass inspected argparse and config/mask implementations; it did not execute this command or install dependencies.

The training CLI writes resolved_config.json, provenance.json, environment.json (including source hashes), validation.jsonl, selection.json, selection_binding.json and best.pt. It does not write a test report. Keep these artifacts out of the source repository. There is no automatic search, subgroup runner, resume facility or cross-run test lock.

## Results ledger

| Study | Dataset/version | Protocol/mask | Seeds | Recall/NDCG | Status |
|---|---|---|---|---|---|
| Stage 1 | Not selected | Not frozen | Planned only | Not available | NOT RUN |
| Stage 2 | Not selected | Not frozen | Planned only | Not available | NOT RUN |
| Stage 3 | Not selected | Not frozen | Planned only | Not available | NOT RUN |

No performance table should contain invented, extrapolated or synthetic-as-real values. A null or negative result is valid evidence. Interpret results alongside the [novelty review](novelty_review.md), and narrow claims if the matched controls explain the gains.


## Separate final test command

After freezing a single run using validation only, execute externally:

```text
python evaluate.py --run /external/selected-run --manifest /external/dataset/manifest.json --execute-final-test
```

The final evaluator checks saved resolved configuration, exact manifest/data provenance, selection metric, SHA256 bindings for configuration/provenance/selection/checkpoint, and checkpoint parameter names/shapes/dtypes/finiteness. It restores the model and calls test evaluation once; it has no optimizer. final_test.json is created exclusively; an existing file is rejected. A failed evaluation after exclusive creation leaves a marker requiring inspection, never silent overwrite. Old runs without selection_binding.json must not be silently upgraded or treated as verified. This binding checks consistency, not protection against a researcher deliberately rewriting all metadata.

Final-test binding includes environment.json. The saved source fingerprint must match current whitelisted source/config contents; documentation-only Git commits are allowed, while missing source checkout evidence or changed source is rejected before evaluation.
