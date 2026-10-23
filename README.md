# dot_rec / BAST

An independently implemented, untrained multimodal recommendation research candidate with two modules: group-conditioned observed transport and selective missing-modality transport on one TRAIN incidence. See [the exact specification](docs/model_design.md), [architecture](docs/architecture.md), and [repository rules](agent.md).

**Research candidate only. GPU training entrypoint has only been statically reviewed, never executed. No training or real-data benchmark has been executed. No performance or novelty improvement is established. Training is prohibited on the current development computer.** Defaults are debugging values, not tuned recommendations.

Minimal dependencies: Python >=3.10, PyTorch >=2.6,<3 and NumPy >=1.24. No PyG, SciPy, pretrained model downloads or paid APIs. The current environment was not modified. Set `PYTHONPATH` to the repository root to use the package without installation.

The CLI exposes training.protocol=mtmt/ftmt. MTMT trains on natural AND experiment visibility; FTMT trains on natural visibility. Both evaluate on the same fixed natural AND experiment mask, recomputing statistics on unchanged TRAIN support. Full-graph projection/propagation repeats each batch; large-scale speed and GPU operation are unverified. Explicit CUDA runs prepare deterministic cuBLAS settings before CUDA initialization; see the protocol for limitations.

Every minibatch recomputes all item projections O(I*(Dv+Dt)*d), full TRAIN CF/transport O(L*E*d+E*d+(U+I)*d), and L2 over ALL trainable parameters O((U+I)*d+(Dv+Dt)*d). This repeats ceil(E/batch_size) times per epoch, with autograd storage in addition to feature/ID/membership storage. Sparse support avoids I² construction but does not establish large-scale speed or memory fit. Do not cache learned embeddings across optimizer steps: that would use stale parameters and change the method.

Provenance and evaluation reports count distinct TRAIN-zero-degree users/items and affected interactions in validation/test. A cold user has p=0, so every candidate score ties and ranks by item ID; this is a defined fallback, **not a solution to user cold-start**. Default evaluation_population=all retains these cases and the full catalog. warm_only explicitly requires positive TRAIN degree at both endpoints of every evaluation target and rejects violations; it never silently drops examples or removes cold catalog candidates.

Safe CPU-only verification (tiny synthetic fixtures, no optimizer steps):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

The user may explicitly execute the following ONLY in a separately authorized environment, after supplying and checking their own local data manifest:

```text
python main.py --config configs/model/bast.json --manifest /external/dataset/manifest.json --output /external/new-run --execute-training
```

Without the explicit flag the CLI refuses to run. Imports do not start training. There is no automatic tuning or background execution. The output directory must not exist. See [data and evaluation protocol](docs/data_protocol.md) before running. This repository does not contain a dataset, model weights, experiment results or a project license chosen on the user's behalf.

Layout: `common` owns dataset, training and evaluation boundaries; `utils` owns incidence/config/mask/sampling/reproducibility; `model` contains BAST and two transport modules. Fixed-LOO/learned-group × compensation off/on are switches in the same model, not separate model implementations.


For reproducibility, environment.json records the commit, dirty flag, and content hashes for Python sources and JSON configs, so local research edits are traceable without blocking dirty runs. The L2 term is a sum of per-parameter-tensor mean squares, not a literal global squared norm. Group-conditioned transport is representable as dynamic bipartite two-step propagation; the hypergraph label itself is not a novelty claim.


Research review drafts: [novelty boundaries and closest evidence](docs/novelty_review.md) and [external experiment plan](docs/experiment_plan.md). Experiments remain unrun; the plan explicitly distinguishes implemented switches from future controls.


Current flat layout follows LGMRec directory responsibilities while using the user-required singular `model/`. Future models belong in `model/`; there is no parallel src implementation. Dependencies are listed in `requirement.txt` and pyproject.toml (no installation performed). Root commands need no installed package when run from this checkout.

Training is now validation-only. After validation-based run selection is frozen, final testing is a separate explicit action in the authorized external environment:

```text
python evaluate.py --run /external/selected-run --manifest /external/dataset/manifest.json --execute-final-test
```

It verifies data/config/checkpoint bindings, never trains, and refuses an existing final_test.json. Both entrypoints refuse execution without their explicit opt-in flag. No real training or final benchmark has been run here.

Verified local tests used PyTorch 2.6.0; the declared >=2.6,<3 range is not a guarantee that every version/device has been tested. Final evaluation also checks the saved source fingerprint against the current source checkout, ignoring Git commit differences alone (for example documentation-only commits). Missing or changed source fingerprints are rejected.

LGMRec data/metric alignment: see [verified contract and limits](docs/lgmrec_alignment.md). The optional configs/model/bast_lgmrec.json reports Recall/NDCG/Precision/MAP at 5/10/20/50 and explicitly filters only cold target users. Three dataset manifest templates require actual file verification and SHA256 replacement; they are not ready-to-run data.
