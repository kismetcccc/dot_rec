# Local verification report

Scope: CPU-only tiny synthetic forward, scalar-reference, data-fixture and finite-autograd checks. No training loop, optimizer step, real dataset, GPU check, convergence experiment or benchmark was executed. Dependencies were not installed or changed.

Command (PYTHONPATH points to src): `python -m unittest discover -s tests -v`.
Latest result: **32 tests passed**. This is implementation evidence, not evidence of recommendation quality or end-to-end training viability.

- 16 model tests: scalar LOO reference, fixed-LOO hand values, paired-donor F1/F2 (.25 → 1/6 → .25), opposite/orthogonal/zero agreement, no paired donors under interleaved missingness, variance and q ablations, attenuation without renormalization, exact full-missing CF fallback, complete-modality inactive compensation, cold/singleton/empty support, hidden NaN isolation, finite gradients at zero norms, residual bounds, 2×2 parameter layout, deterministic scores.
- 10 protocol tests: split overlap/duplicates, reproducible masks, TRAIN-only negative sampling, chunked ranking and tie/mask rules, hand-computed metrics, local manifest/hash/MMRec/CSV ingestion, unknown configuration keys and CLI refusal without opt-in; MTMT/FTMT training/evaluation visibility and finiteness boundaries; CUDA determinism preflight with entirely mocked environment/state (no CUDA initialization or actual environment change); hand-counted cold endpoint/interaction coverage, warm_only rejection without filtering and zero-user item-ID tie behavior.
- 6 API tests (multiple cases each): negative/out-of-range/float/bool/non-vector IDs; paired/loss shape mismatch and empty loss; valid int32/int64 and empty prediction; float32/64 finite zero normalization; explicit rejection of zero FP16/BF16 and CPU AMP; CPU construction independent of default floating dtype; view-mask conversion; encoded tensor shape/device/dtype contract; strict finite non-boolean numeric configuration and positive-integer catalog dimensions.

The first file-fixture attempt encountered Windows owner-only temporary-directory ACL restrictions. Tests now create isolated UUID scratch directories under ignored .cache/unit-tests, check the resolved cleanup parent and remove only their own fixture directory. The two empty original scratch directories were checked and removed. The rerun passed; no unresolved temporary-file blocker remains.

Public execution contract: model construction is CPU/float32, followed by whole-model .to(device) or .double(). IDs and view masks are validated and moved safely. Only float32/64 execution is supported; AMP and half/BF16 are rejected, not silently treated as supported. Device-transfer policy has CPU tests, but actual GPU execution is unverified.

Remaining validation boundaries: training/checkpoint workflow was statically reviewed but not executed; performance, scalability, data-owner ID mapping correctness and novelty remain unverified. The test suite must not be extended to run training on this development computer.


Latest review verification: 34 CPU synthetic unittest methods passed (2026-10-23); compileall and git diff --check passed. Added rejection tests for duplicate/noninteger cutoffs, invalid batch/chunk sizes, negative/floating user IDs, and ranked/user row mismatch. Added source fingerprint scope, deterministic repeat, and content-change tests using synthetic files only. No training, optimizer steps, CUDA execution, or real-data benchmark performed.
