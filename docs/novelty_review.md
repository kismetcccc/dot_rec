# BAST novelty review: a defensible candidate, not a priority claim

Status: documentation review for implementation commit `9bb81bbe53d4a5b4c3da207c49163ae41caf6e98`. No real training results exist. Evidence scope as of 2026-10-23: static literature and code review, without reproduced experiments. This document does not imply all linked sources were re-fetched on that date. Pinning a repository identifies the reviewed snapshot, not a license to copy it. This is not an exhaustive literature review.

## Closest evidence and boundaries

| Work and evidence | Relevant prior idea | Boundary for BAST |
|---|---|---|
| MMHCL (2025), [paper](https://arxiv.org/abs/2504.16576), [author repository snapshot](https://github.com/Xu107/MMHCL/tree/0ceaa26ae637a65c2043a01899fbed5608428b0b) | Multiple u2u/i2i/UI paths and contrastive learning | Multimodal hypergraph recommendation is established; BAST uses one TRAIN support and no contrastive objective. |
| OrthoRec (September 2026 preprint), [paper](https://arxiv.org/abs/2609.02152), [author repository snapshot](https://github.com/Camilla-jl/Orthorec/tree/631138035a562db09ac8931d113ff41d27db07dd) | CF anchoring, orthogonal purification, independent sigmoid routing and safe SSL | CF anchors and sigmoid reliability are not claimed new; BAST does not add orthogonal purification or SSL. |
| MARGO (2025), [paper](https://arxiv.org/abs/2504.16524), [code lines 195-215](https://github.com/hello-dx/MARGO/blob/7140601472e76ba4456e2afc94997544f375b912/margo.py#L195-L215) | BPR-margin pseudo reliability labels and confidence KL | Reliability supervision already exists. BAST q is support/agreement attenuation, not a calibrated probability or margin-supervised confidence. |
| I3-MRec (MM 2025), [paper](https://arxiv.org/abs/2508.04247), [author repository](https://github.com/HuilinChenJN/I3-MRec) | Missingness simulation with IRM/IB | Modality dropout is not novel. BAST does not implement IRM/IB. Repository SHA was not supplied in this review. |
| CaIRec (July 2026), [paper](https://arxiv.org/abs/2607.26720) | Structural completion and recommendation-space calibration | Completion/calibration are prior themes. Official code has not been verified; no third-party implementation is labeled official. |
| GRE-MC (SIGIR 2026), [paper](https://arxiv.org/abs/2605.00670) | Anchored retrieval, subgraphs, Transformer and codebook | Anchored completion is not itself new. Official code has not been verified. BAST uses no retrieval expansion, Transformer or codebook. |
| HIRE/HIRES (TOIS 2025), [author page](https://www.cs.emory.edu/~jyang71/), [author full text](https://www.cs.emory.edu/~jyang71/files/hires.pdf), [DOI](https://doi.org/10.1145/3745020) | Completeness and interaction-support priors with sparse OT hypergraph completion | Support/completeness priors and hypergraph completion are prior art. Official code is unverified. The May 2024 template date must not be used as the publication year. |
| HyperFuse (October 2, 2026 preprint), [paper](https://arxiv.org/html/2610.03211v1), [code snapshot](https://github.com/ideas-are-bulletproof/HyperFuse_benchmark/blob/121df1ca13b17c5b7b469b93d2bb791890674dd7/vendor/fuse/fuse_versions/HyperFuse.py) | Lines 376-397: group cosine/mean/std/count/log-count; 477-497: mean-one bounded utility; 620-633: renormalized propagation | Group statistics and count-dependent gating are not new. BAST keeps original count denominators to retain attenuation. HyperFuse is not a recommendation benchmark: do not compare its scores with Amazon recommendation metrics. |
| HypeBoy (ICLR 2024), [paper](https://arxiv.org/abs/2404.00638), [author repository](https://github.com/kswoo97/hypeboy) | Hyperedge prediction excluding the target | Leave-one-out is not claimed first. Repository SHA was not supplied in this review. |

See also [framework source review](research_sources.md) for pinned MMRec/ChaoRec interfaces, selection risks and license boundaries. No third-party implementation was copied into this candidate.

## Narrow candidate distinction

The hypothesis concerns this exact combination, not any individual building block:

1. One fixed binary TRAIN incidence supports both modules. Per-incidence leave-one-out group statistics condition observed-modality attenuation; the received donor message also excludes the target itself.
2. Original member counts and item degrees remain the denominators. Gate mass is not renormalized away, so weak evidence can contribute a small or zero message.
3. Missing-only transport uses the same paired, currently fully visible donors for both modality means on that same support. It operates only when the other modality is visible, and completed content never feeds back into donors, CF, or support.

At any forward pass, weighted-incidence transport remains expressible as dynamic bipartite two-step propagation. Do not claim that pairwise graphs cannot express it, or that the word hypergraph proves higher-order novelty. Group-conditioned weights can differ from a fixed linear kernel without establishing a new general class of computation.

The reviewed evidence has not identified an identical full construction. That is not proof of priority or exhaustive novelty: the combination may be incremental. A defensible current description is an independently implemented candidate combining group-conditioned attenuation and selective missing transport on a shared behavioral support. Performance and novelty remain hypotheses.

## How the hypothesis could fail

Learned gates may only shrink messages, rather than encode useful group structure. Histories can contain legitimate multiple interests; variance is not necessarily noise. Angular collapse can lower variance without improving ranking. Two gates in observed transport can over-attenuate. Paired donors disappear under high or interleaved missingness; low q cannot manufacture useful evidence. Complete-modality input disables module 2. Cold users have zero collaborative vectors and tie scores, not a learned cold-start solution.

Matched constant gates, shuffled gates/donors, q ablations, and subgroup/coverage diagnostics should test these explanations before stronger claims. Do not add mechanisms merely because related work exists. See [experiment plan](experiment_plan.md); every result is currently unrun.
