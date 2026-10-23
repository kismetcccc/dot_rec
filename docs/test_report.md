# Local verification report

Latest result: **42 tests passed** (16 model, 13 protocol, 6 API, 3 final-entrypoint checks, 4 LGMRec alignment checks). Earlier 32/34 counts are historical snapshots, not current results.

CPU-only tiny synthetic forward, scalar reference, local synthetic data fixtures, finite-autograd checks, mocked final-evaluation control flow and static training-call inspection. No training loop, optimizer step, real dataset benchmark, GPU execution or dependency installation.

From repository root: python -m unittest discover -s tests -v. Compile check: python -m compileall -q model common utils tests main.py evaluate.py. git diff --check passed.

Coverage: LOO references, paired donors F1/F2 and A/B/A masks, hidden NaNs, exact CF fallback, cold/singleton/zero norms, attenuation bounds, ablations, masks, split/hash validation, TRAIN-only sampling, ranking ties/masks, MTMT/FTMT and cold coverage. Ranking rejects duplicate cutoffs/valid IDs, floating ranked IDs, negative targets and malformed/out-of-range edges; repeated -1 padding never hits. API rejects broadcasting and half/AMP.

Final-test tests use AST call inspection and stubbed CLI dispatch, never training. A synthetic checkpoint with mocked loader/model/evaluator tests artifact mutation, parameter/provenance mismatch, single test call and overwrite refusal. CUDA setup is mocked, without hardware access.

Flat-layout equivalence passed: seed 7 tiny CPU outputs p/v/behavior/z/h/c/g/q exactly match saved pre-move tensors with torch.equal. BAST, two transport modules and incidence ASTs match HEAD ignoring imports only. Both root CLI help commands and missing opt-in rejection passed before data loading.

Fixture directories and pre-move tensors are ignored synthetic cache, never publishable artifacts. GPU execution, end-to-end optimization, scalability, dataset validity and novelty remain unverified.

Final-entrypoint regression cases also mock a changed source fingerprint and an unavailable fingerprint: both reject before evaluating. environment.json is bound by SHA256. These additional subcases remain in the final-entrypoint tests.

LGMRec regression checks cover MAP min(K,P), Precision/K, macro versus micro, short rankings/padding, 16-metric config propagation to validation and final evaluation, explicit cold-user-only filtering without removing cold items or mutating support, and rejection of manifest hash placeholders before data access.
