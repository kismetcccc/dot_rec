# BAST: frozen candidate specification

BAST means Behavior-Anchored Selective Transport; the short name is provisional, not a uniqueness claim. This is an independently written candidate, not a reproduction of the user's unavailable code. No training, convergence, performance, or novelty claim has been validated.

## Dataflow and support

Manifest → data validation → TRAIN edges → one binary item-by-TRAIN-user incidence H → shallow collaborative encoder and two modality modules → item representation → dot-product score → independent ranking evaluator. Validation/test edges never enter the model, support, statistics, or negative sampler. Natural availability AND the fixed experiment mask defines evaluation visibility. Training-only modality dropout acts on that visibility BEFORE both modules compute statistics, using one shared view.

The explicit training.protocol switch selects mtmt or ftmt. MTMT uses natural AND experiment visibility for the training base; FTMT uses natural visibility for training (full means all naturally available modalities, not revealing naturally absent data). Both use fixed natural AND experiment visibility for validation/test, with fresh statistics on the unchanged H. The loader validates all training-visible features for finiteness; FTMT cannot hide a NaN merely by dropping it from the evaluation mask.

H has one column per user (empty columns are harmless); each TRAIN pair appears once. No modality adds an edge. There is no KNN graph, second hypergraph, diffusion, teacher, or iterative imputation. Sparse memberships are arrays (item, anchor); operations use index_add, never H Hᵀ or a dense item-item matrix. The two-way normalized TRAIN adjacency also provides an ordinary shallow LightGCN base (p_u,b_i), averaged over layer 0 through L. Zero-degree rows are zeroed including layer 0: random untrained IDs are not cold-start information. LightGCN is a conventional baseline, not the contribution.

Fixed raw features x_i^m are sanitized by where(o_i^m,x_i^m,0) BEFORE bias-free projection, safely L2-normalized, then masked again, giving z_i^m. Observed nonfinite features are an error; hidden NaNs are removed. Safe cosine returns zero when either norm is zero.

## Module 1: group-conditioned observed transport

For each anchor e and modality m, using only visible members:

    n_e = Σ_i H_ie o_i
    S_e = Σ_i H_ie o_i z_i
    T_e = Σ_i H_ie o_i ||z_i||²

For membership (i,e): nlo=n_e-o_i; μlo=(S_e-o_i z_i)/max(1,nlo);
varlo=max(0,(T_e-o_i||z_i||²)/max(1,nlo)-||μlo||²).
The behavior LOO mean uses unit-normalized b_j and ALL TRAIN members except i, independently of modality availability.

Only if o_i=1 and nlo>0:

    g_ie = sigmoid(θ0 + softplus(θB) cos(b_i, μB_lo)
                         + softplus(θC) cos(z_i, μlo)
                         - softplus(θV) varlo
                         + softplus(θN) log(1+nlo))

Otherwise g_ie=0. There are exactly five learned scalars per modality, no MLP. The matched fixed-LOO ablation replaces eligible gates by 1; it retains the same parameters and every other setting.

    G_e = Σ_j H_je o_j g_je z_j
    t_i = o_i / max(1,d_i) · Σ_e H_ie g_ie
          (G_e - o_i g_ie z_i) / max(1,nlo_ie)
    h_i^m = o_i z_i^m + η t_i^m

d_i=Σ_e H_ie. Empty-LOO messages are explicitly zero, including singleton anchors. Denominators are ORIGINAL counts, never gate sums. Fixed histories plus linear propagation form a two-hop kernel; calling this a hypergraph is not itself novelty. The specific difference under study is group-conditioned nonlinear membership weighting.

## Module 2: missing-modality selective transport

Only targets missing m but observing the other modality n are eligible. Donors for BOTH means must be the SAME TRAIN members with BOTH modalities visible in the current view. A completed representation never becomes a donor.

For each anchor, compute paired-donor count k, means μ_m and μ_n, and variance var_m from unit donor z. For each eligible target membership:

    q_ie^m = k_e/(k_e+κ) · exp(-var_e^m/τ)
             · relu(cos(z_i^n, μ_e^n))
    c_i^m = (1-o_i^m)o_i^n / max(1,d_i)
             · Σ_e H_ie q_ie^m μ_e^m

q=0 when k=0. H is binary with uniform donors, so effective donor count here is count, not a separate learned effective sample size. No normalization by Σq is allowed: weak evidence must stay weak. No biased transform, feedback to CF/support/donors, or second transport round. This is support/consistency attenuation, not probability calibration or a causal effect.

Since 0≤q≤1 and ||μ||≤1, ||c_i^m||≤1. With two modalities at most one missing branch is eligible; hence ||βΣ_m c_i^m||≤β. Both missing gives exactly b_i; zero-degree targets have no transport; cold items with content retain their own projected content. These are implementation properties, not guarantees of good ranking.

## Score, loss and ablations

    v_i = b_i + α Σ_m h_i^m + β Σ_m c_i^m
    score(u,i) = p_uᵀ v_i
    loss = mean softplus(score_neg-score_pos) + λ L2

η,α,β are explicit coefficients in [0,1], defaults .2,.2,.1 ONLY for debugging; κ=2, τ=1, L=1, d=64 are also unvalidated defaults. L2 is the sum of mean squared entries of each trainable parameter tensor (ID embeddings, projection weights, ten gate scalars); raw features are fixed buffers. No CL, reconstruction, IRM, IB, teacher, or auxiliary loss. One optimizer stage is provided for user execution elsewhere, never run on this computer.

The primary 2×2 is fixed-LOO vs learned-group gates × compensation off/on. Parameter allocation and all unrelated defaults are identical. q ablations: full, unit (q=1 for nonempty eligible donors), no_support, no_agreement. Nonempty/target-eligibility guards remain in every ablation. Do not silently change masks, data splits, CF depth, or loss weights between variants.

## Evaluation and evidence limits

All split duplicates and cross-split pairs are rejected by the data boundary; the graph builder independently deduplicates TRAIN edges. Negative sampling excludes TRAIN positives only. All-catalog chunked top-k excludes TRAIN history in validation and test; optional exclusion of validation history at test is explicit in config (default false, matching TRAIN-only masking). Validation alone selects checkpoints; a separately invoked evaluate.py verifies and loads the selected checkpoint for one final test; training itself never evaluates test. No automatic hyperparameter search is provided; manual runs must also select by validation alone.

Fixed masks, hashes, resolved configuration, seeds, manifest declarations, environment versions and code revision are recorded. Unknown metadata must remain an explicit user declaration, never an inferred Amazon version. Synthetic CPU tests and finite-gradient checks do not establish training viability, convergence or benchmark performance.

Research risks: user histories can mix interests and these interests are not necessarily noise; low variance is not ranking quality, and angular collapse can imitate low variance; paired donors can be scarce; module 2 is inactive under complete modalities; the two gates in module 1 can attenuate too strongly. No extra preventive losses are added. Behavior anchoring and each constituent are not individually claimed new. The candidate claim is the exact combination of group-conditioned propagation and rejectable missing transport on one behavior support; exact novelty and effectiveness remain unproven.

Further research controls (not additional framework modules): compare learned gates to a constant gate matched for average attenuation and to shuffled gates, with support, masks, coefficients and loss fixed. These can distinguish useful group conditioning from generic shrinkage. Only the requested fixed-LOO 2×2 and q switches are implemented now.

For default coefficients, triangle bounds on semantic residual are .48 for both observed, .34 for one observed plus possible compensation, and exactly 0 for both missing. Tests verify these bounds and the individual ||t_m||≤1, ||c_m||≤1 claims on small inputs. Interleaved single-modality missingness can leave 50% visibility per modality yet no paired donors: compensation must be zero in that case. Incidence accepts only binary memberships from integer pairs; weighted incidence is not supported.

Related-work pointers (official-code status is limited to the checks documented in novelty_review.md): [MMHCL](https://github.com/Xu107/MMHCL), [OrthoRec](https://github.com/Camilla-jl/Orthorec), [MARGO](https://github.com/hello-dx/MARGO), [CaIRec](https://arxiv.org/abs/2607.26720), [GRE-MC](https://arxiv.org/abs/2605.00670). In particular, this task has not verified official code for the last two. These pointers motivate novelty checks; no third-party implementation is copied. Framework evidence and licensing boundaries are separately recorded in research_sources.md.


At each forward pass, weighted-incidence propagation remains expressible as two-step message passing on a dynamic bipartite graph. We do not claim that pairwise graphs cannot express it or that the hypergraph name establishes higher-order novelty. The investigated distinction is the group-conditioned weighting and its exact combination with selective missing transport.

The precise regularizer is `L2 = sum(parameter.square().mean() for parameter in model.parameters())`, not the unnormalized sum of squared entries across all parameters. Each tensor contributes its own mean; this weighting matters when comparing implementations.

Run environment metadata includes a SHA256 source fingerprint and per-file hashes of regular `model/common/utils Python plus main.py/evaluate.py` and `configs/**/*.json` files, including local modifications. Symlinks are excluded. No external datasets, .env files, or runtime outputs are hashed. Resolved configuration is recorded separately; an empty source scope is reported with a null digest rather than claiming reproducibility.
