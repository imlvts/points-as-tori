# Training improvements

Changes to the training code of this repository (MIT, (c) 2026 Nicole Feng), from a study of the trainer in
November 2026. Finding ids (F*, X*) refer to that study's notes. A second round (November 2026) applies the
suggestions of an external review of the trainer (`TRAINING_SUGGESTIONS_GEMINI.md`, not part of the repository);
section numbers § refer to that review. Changes that are exact, or measured better in every respect, are always on;
everything else is a flag, off by default. The default architecture (`FundamentalFormPredictor`) and the default
`python train.py` are unchanged; the architecture variants of the second round are options (`--frame`,
`--input-mlp`, `--encoder`, `--wls`, `--aux-heads`), saved with the model (key `'arch'` of the .pkl, only when not
default) so that `eval_model.py`, `infer.py` and `--init` restore them.

## Bug fixes (always on)

| fix | finding | effect |
|---|---|---|
| Guard-free log-sum-exp form of the Eq. 25 blend (`network.py`, `torus_distance_gradient_masked`) | F2 | The blend took sigma/lambda from the 64 nearest points but summed over all points and divided by `max(sum W, 1e-10)`; for queries beyond ~0.9x the 64-NN radius every weight underflowed and the trainer saw phi ~ 0 with a meaningless gradient, which rewards over-curved tori. Same value wherever the guard did not fire. |
| Geometric sign rule: torus center from kappa_min (`torus_precompute_masked` and the C++ `fit_torus_at_point`) | F4 | At saddles the center was put on the wrong side (the mirror image of the osculating torus, dg/dn* = -1 at the point), a loss cliff at kappa_min kappa_max = 0. Non-saddle tori are bit-identical; with the released weights 1 of 4096 snail tori and 35 of 12917 holey-bunny tori change. Training and inference use the same rule. |
| Padded shapes give finite, correct gradients (`InputFeatures`, `torus_precompute_masked`, loader, train loop) | F3 | Padded rows (neighbors = -1) had a zero neighborhood scale, 0/0 = NaN features and an all-NaN gradient that was zeroed: about 45 % of the simulated ABC scans gave no training signal. Now the gradient of a padded shape equals that of the same shape without padding (rel. 3.5e-5); batches of padding only are skipped. |
| Local zero-set / narrow-band query sampler (`mesh_sampling_jax.py`) | X9a | Zero-set and band queries were sampled on whole faces whose bounding box meets the neighborhood box; on CAD meshes 32-87 % of them were outside the neighborhood. Now rejection sampling on mesh ∩ box (0 % outside). |
| Unit normals on load (`load_mesh_training_data`) | F1 | `pretraining_blob.npz` and the validation files store normals of length 2, which made over-curvature free in pretraining and the validation numbers meaningless. |
| Singularity-free frame form of the torus SDF in the training blend (`torus_frames_masked`, `torus_signed_distance_frame`, `torus_distance_gradient_masked`; a82c5cc) | §1.1, §1.2 | The blend built explicit tori in float32 (center p* + n*/kappa_min, radii from 1/kappa, axis through (theta, phi)) and differentiated through 1/kappa. Nearly flat tori cancel: |kappa_min| sigma < 1e-3 for 1.5 % of the L13 tori; float32 errors of the explicit torus SDF up to 32 (relative to max(1, |g|)) with sign flips. Now each torus is a contact point, a frame (v_min, axis, n*) and (kappa_min, kappa_max), evaluated in the rationalized form without division by kappa. Same function in exact arithmetic (float64: <= 4e-15). Training blend on L13 weights, error against float64, p99.9 / max: 7.0e-4 / 2.1e-2 sigma before, 1.1e-6 / 4.8e-5 after; float32 error of the loss gradient w.r.t. the coefficients up to 580 % before (one shape), <= 1.2 % after. Changes training: see "Measurements" below. The C++ evaluator (double, explicit form) needs no change. |
| Sign-preserving epsilon guards (+1 at 0, as the C++ `sign()`) in the torus fit (fecea25) | §1.2 | `jnp.sign(0) = 0` turned a quadric with v2 = 0, v1 < 0 into a degenerate torus (R = 1e10) with NaN gradients. On real data 0 (released) and 1 (L13) of 30160 tori change. |
| Finite derivative of \|grad phi\| at 0; NaN distances dropped with `where` instead of multiplied by 0 (`SDF_loss`, `validation_loss`; c976d5a) | §1.2 | sqrt'(0) was NaN, and NaN * 0 = NaN. Identical values wherever finite. In 1200 checked training steps (CPU, before and after, random and L13 initialization) no step had a non-finite gradient either way; the §1 fixes improve gradient accuracy, not NaN counts. |
| `InputFeatures` scales the offsets p_j - p_0 instead of subtracting scaled positions (ac24280) | §2.1 | float32 cancellation on clouds far from the origin: released weights, cloud translated by 1000, median max relative coefficient change 1.1e-3 before, 6.3e-4 after. The training data are centred; this matters mostly at inference. |
| Validation in chunks of 128 rows (002c387) | | The end-of-phase validation generated the queries of a whole 2048-point shape at once: 35.6 GiB of temporaries (XLA estimate, CPU), now 2.2 GiB; loss with the all-points blend 11.3 -> 1.5 GiB. Same loss to rel. 1e-7. |
| Periodic validation of the patch loop recorded in the history (d9c21b8) | §4.2 | It was printed but not stored. |
| Training history plot saved as `MODELS_DIR/<model>_history.png` and closed (7b72334) | §4.2 | It was `PlotTrainingHistory.png` in the working directory: every phase and every run from the same directory overwrote it. |

Checked and left unchanged: the phase-1 learning rate is 1e-5 in the code but 1e-6 in the paper (Sec. 4.3); it is
1e-5 in every commit since the first one, from which the released weights come. `next_shapes()` draws from the global
numpy RNG, which is seeded at the start of every phase and used by nothing else in the patch loop: the shape stream is
deterministic per seed (§4.2's third bookkeeping item is not a bug); changing it would change the stream for no gain.

## Speedups (always on; same outputs up to float32 rounding)

| change | finding | effect |
|---|---|---|
| The last Transformer block computes attention and FFN for token 0 only (`TransformerBlock.call_first_token`; bca0898) | §3.2 | Coefficients max abs diff 7.6e-6; CPU forward -8 %; XLA estimate (CPU) of the temporaries of the L13 step 14.47 -> 12.83 GiB. Parameters and saved models unchanged. Not with `--wls` (it needs all tokens). |
| Query generation of all data patches of a step in one vmapped call (93a69c0) | §4.4 | Bit-identical queries and distances; one dispatch instead of S. With `--mix`, one compilation per number of data patches in a step. |
| k-NN of the `--noise` / `--quantize` sigma estimate on 8 threads (6dcb048) | §4.4 | Identical values; `prepare()` of 8 shapes with `--noise 0.1` 270 -> 109 ms (it exceeded the 0.21 s GPU step before). |

GPU step time of the screening runs below (6 shapes x 32 centres, the desktop sharing the GPU; not a controlled
benchmark): 0.17-0.20 s/step before the review's changes (fa5cead), 0.15 s/step after (7b2494e).

C++: `TorusDistanceField.set_point_weights` (per-point blend weights; empty by default = unchanged results) for
`--aux-heads` inference. The bindings installed in `PointsAsTori-build/.venv` are the authors' build: they have neither
`set_point_weights` nor the F4 sign rule; rebuild the bindings from this branch (`infer.py` falls back to
`set_outliers` with a threshold and says so).

## Options of `train.py` (all off by default; `python train.py` runs the authors' curriculum)

| flag | finding | what it does |
|---|---|---|
| `--data-dir`, `--models-dir`, `--validation-file`, `--seed`, `--only-phases`, `--max-steps`, `--log-every` | | paths, seed, a subset of the curriculum, a step cap, loss logging |
| `--knn-blend K` | F2/F8 | blend the tori of the K nearest points of each query, the rule of the accelerated inference (K = 32 reproduces the C++ evaluation to 1e-5) |
| `--phase NAME:GLOBS:STEPS` (repeatable), `--init random\|PATH`, `--skip-pretraining` | F1, X2 | a custom curriculum (each phase starts from the previous one), the initial weights, no single-blob pretraining |
| `--min-points N` | F3 | drop shapes with fewer valid points |
| `--lr`, `--warmup`, `--schedule cosine` | X3 | learning rate, warmup and a real cosine decay to 1 % over each phase (the authors' schedule decays over 1e8 steps) |
| `--clip G`, `--weight-decay W` | | gradient clipping by global norm; AdamW weight decay (default 0, as before) |
| `--eik-w W` | X3 | weight of the eikonality term (default 1) |
| `--multishape S --patch-c C --upad U --save-every N` | X9 | each step trains on one patch of C clustered centres from each of S shapes (union of their k-NN, padded to U) instead of rows of one shape |
| `--resample MIN:MAX`, `--resample-match S` | X9 | fresh points on the ground-truth mesh at every visit (count among the powers of two in [MIN, MAX]), k-NN recomputed; only for files whose path contains S |
| `--rotate` | X9 | random rotation of every shape visit |
| `--noise S`, `--quantize Q` | | Gaussian noise of S sigma on the input points; quantization of the input points along a random direction to steps of Q sigma (normals and ground truth stay exact; not evaluated yet) |
| `--mix analytic:FRAC:PATH` | X12, X13 | a fraction of the patches from analytic scenes with an exact SDF (`gen_analytic.py OUT.npz N SEED`) |
| `--gate`, `--no-gate` | | evaluate every phase's model with `eval_model.py` and print the gate checks (log only); `--no-gate` drops the gate of a preset |
| `--preset l13` | | our best recipe (L13) in one command; explicit flags override it |

`--resample`, `--noise`, `--quantize` and `--mix` need `--multishape` (the authors' loop generates queries for every
point of a shape and uses the stored neighbors). So do all options of the next table except the architecture options.

### Options from the review (all off by default)

The last column is the screening result against the same code without flags (section "Measurements"): gain,
mixed, none (no measurable effect at 9000 steps), worse, fails.

| flag | § | what it does | screening |
|---|---|---|---|
| `--frame continuous` | §3.1 | network input in Frisvad's tangent frame (continuous except at the direction -(1,2,3)/sqrt(14)) instead of the authors' frame, which jumps at \|n_x\| = 0.9; prediction rotated back exactly | mixed (1 seed) |
| `--frame invariant` | §3.1 | weighted average over three frames attached to the neighbourhood (spin-1 and spin-2 moments of the tangent offsets): exactly rotation-invariant, 3 network passes (training: one after another under remat). Also at inference and evaluation of any weights: `PointsAsTori(frame=...)`, `eval_model.py --frame` | gain (3 seeds), 3.3x step time |
| `--input-mlp` | §3.2 | 13 geometric features per neighbour (local coordinates and normal, distances, quadratic terms, secant curvatures) through Linear-GELU-Linear-LayerNorm | mixed (1 seed) |
| `--encoder perceiver [--latents 8 --cross-layers 2 --self-layers 2]` | §3.2 | pointwise MLP on the 64 neighbours, cross-attention into 8 latent tokens; 7x fewer network FLOPs | worse alone (1 seed); see `--wls` |
| `--wls [--wls-bound B]` | §3.2 | the six coefficients are a weighted least-squares jet fit (heights and normals) with learned per-neighbour weights, plus a residual bounded by B median distances (default 1; 0: pure WLS) | with `--encoder perceiver`: gain (3 seeds) except the far field; B = 0 fails |
| `--aux-heads` | §3.3 | two more outputs per point: a normal-flip logit (the torus SDF is negated) and a blend confidence (log weight added in the blend; at inference through `set_point_weights`) | fails (training diverges) |
| `--corrupt-flip P`, `--corrupt-outliers P` | §3.3, §4.3 | negate each input normal with probability P; move each input point with probability P by 1-5 sigma in a random direction, with a random normal | gain on corrupted input, mixed on clean data (1 seed) |
| `--band-sigma B`, `--min-extent-sigma M` | §2.1 | narrow-band offsets U(-B, B) sigma_c and minimum query-box extent M sigma_c (sigma_c: the centre's median 64-NN distance) instead of 0.2 and 0.1 world units | none |
| `--dist-sigma REF` | §2.1 | distance error in units of sigma_c, times REF (REF keeps the loss size; 0.125 for `--resample 1024:8192`) | none |
| `--normal-w W`, `--normal-band` | §2.2 | add W mean(1 - cos) of grad phi against the true normal at the zero-set queries (and at the band queries whose true gradient is known) | gain (3 seeds) |
| `--lipschitz-cube` | §2.3 | one-sided max(0, \|grad phi\| - 1) on the cube queries instead of \|1 - \|grad phi\|\| | none |
| `--aux-curv W` (needs `--mix`) | §2.4 | add W times an L1 loss on all union points of the analytic patches: atan of mean curvature and anisotropy times sigma, contact offset / sigma, tilts | mixed (3 seeds) |
| `--schedule wsd`, `--wsd-decay F` | §4.2 | one warmup-stable-decay schedule over all phases: constant lr, cosine to 1 % over the last F of all steps (default 0.2) | gain with `--carry-opt --ema 0.999` (3 seeds) |
| `--carry-opt` | §4.2 | carry the AdamW state (moments, step count) from phase to phase (`<model>_opt.pkl`) | none alone |
| `--ema D` | §4.2 | exponential moving average of the weights; the EMA weights are validated, gated and saved as the model, the raw ones as `<model>_raw.pkl` | none alone |
| `--normal-noise DEG` | §4.3 | tangent Gaussian noise of DEG degrees per component on the input normals | none on clean data (3 deg) |
| `--holes P:R` | §4.3 | with probability P per patch, remove the input points of a ball of radius R sigma next to the seed | none (1 seed) |
| `--seed-area` | §4.3 | patch seeds drawn with probability ~ s^2 (s the 8th-neighbour distance, ~ surface area per point) | none |
| `--resample-decimated MAX` | §4.1 | meshes decimated to the face limit (2048 faces) are resampled to at most MAX points | none |
| `--resample-analytic MIN:MAX` | §4.1 | point counts of the analytic scenes of `--mix` (default: those of `--resample`) | none |
| `--union-k K2` | §4.4 | the patch union also holds the K2 nearest points of every centre (halo, so that the patch blend sees the points the inference blend sees) | worse (K2 = 96) |

## Our recipe (L13)

```
cd training
python train.py --preset l13 --data-dir /path/to/PAT_training_data --models-dir models/l13
```

Phase 1: nicely sampled blobs + ABC, 150k steps from random initialization; phase 2: the same plus the simulated
scans (at least 128 points), 70k steps. 8 shapes x 32 centres per step, 32-NN blend, resampling 1024-8192 points
(nice shapes), rotation, AdamW lr 1e-5, 1000 warmup steps, cosine to 1 %, clip 1, weight decay 0, loss L1 distance +
0.1 L1 eikonality, seed 1. About 13.5 h and 14 GB of GPU memory on an RTX 4080. Set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.97`.

The L13 weights were trained with the code before the review's changes. The preset is unchanged, but the always-on
changes above change its training (see "Measurements"); it has not been rerun at full length with them. The 14 GB were
measured before the token-0 readout, which lowers the XLA estimate of the step by 1.6 GiB (CPU). On the 16 GB card
with the desktop running, 8 shapes still do not fit: the step needs a 12.72 GiB buffer with the current code (14.25
GiB before the review). The preset keeps 8 shapes (for a GPU without a desktop or with more memory); the screening runs
below use 6.

Check (1500 steps of this recipe on blobs + ABC17 with 100 warmup steps, seed 1, against our original trainer's
equivalent runs; the patch stream is the same, so the per-step losses nearly coincide: final loss 0.01548 vs 0.01551;
code and `eval_model.py` before the review, i.e. the old blend form and the all-points field blend):

| | kerr | sphere k rho | analytic angle | far p99.9 | snail p99.9 |
|---|---|---|---|---|---|
| this trainer | 0.057 | -0.76 | 3.40 deg | 0.27 | 0.68 |
| our trainer, same seed | 0.056 | -0.76 | 3.37 deg | 0.29 | 0.68 |
| our trainer, 3 seeds | 0.036-0.063 | -0.71..-0.92 | 3.10-3.40 deg | 0.23-0.29 | 0.68-0.69 |

## Evaluation

```
python eval_model.py models/l13/FundamentalFormPredictor_p2.pkl --data-dir /path/to/PAT_training_data --gate
```

About 2 minutes per model: curvature tracking on analytic clouds (kerr, sphere k rho, anisotropy, saddles), normal
angle and level-set curvature at fresh surface samples, error tails of the blended field (sphere, torus, plane,
cylinder, `data/block`, `femur`, `knot`, the snail) and the loss on fixed validation queries; the gate checks
kerr <= 0.1, analytic angle <= 4 deg, sphere k rho within 30 % of -1, plane k sigma < 0, snail outward p99.9 < 1 sigma,
far-field p99.9 < 0.5 sigma.

Since the review (0ca3e6b) the field metrics (normal angles, level-set curvature, band, far-field and snail tails)
use the blend over the 32 nearest points of each query, the field of the accelerated inference; `--blend-knn 0` gives
the all-points blend of earlier versions. Numbers in the columns angAn, angMe, lsH, b999, bmax, far999, sn999 from
earlier versions (including the check table above) are not comparable with the current ones. For the released
weights the 32-NN field is slightly worse at the surface (analytic angle 3.72 -> 3.93 deg, lsH 0.556 -> 0.584) and the
same in the tails. The field also uses the frame form of the blend (float32 differences against the old form up to
7e-4 sigma at p99.9 for L13-like models with nearly flat tori). `--frame legacy|continuous|invariant` evaluates the
weights in another tangent frame.

The gate check "plane k sigma < 0" is obsolete: it encoded the curvature offset that the old blend form gave every
model on flat input. With the frame form the plane curvature is k sigma ~ +-0.01 with a random sign (nbase seeds
-0.013, -0.013, +0.012), so its MISS is noise; it was already seed-dependent before (base seed 3: +0.038, MISS). The
check is still printed; read a MISS there as "|k sigma| small".

## Measurements of the review's suggestions (November 2026)

Raw material (runs, logs, scripts): `ddg-tools/runs_tmp/pat-gemini/` (not part of this repository).

### Protocol

Screening, shortened L13 preset: phase 1 nicely sampled blobs + ABC, 6000 steps from random initialization; phase 2
the same plus the simulated scans, 3000 steps; warmup 300 steps, cosine to 1 % of the lr in each phase; 6 shapes x 32
centres per step (8 do not fit on the 16 GB RTX 4080 next to the desktop); otherwise as L13 (upad 256, 32-NN blend,
resampling 1024-8192, rotation, lr 1e-5, eikonality weight 0.1, clip 1); no validation during training. About 26 min
per run. Then `eval_model.py` (32-NN field) on the phase-2 model. `--mix` runs use 25 % analytic patches from 6000
scenes (`gen_analytic.py`, seed 1).

- `base`: the code before the review (fa5cead), seeds 1-3.
- `nbase`: this code (7b2494e) without flags, seeds 1-3: the reference for every flag.
- Every flag: seed 1; variants outside the nbase band repeated with seeds 2 and 3.

Columns (`eval_model.py`): kerr (mean curvature error x sigma, analytic clouds), slope (predicted vs true H sigma,
ideal 1), aniso (anisotropy slope, ideal 1), inSad (fraction of torus inner-ring points predicted as saddles, truth 1),
angAn / angMe (normal angle of the field at analytic / mesh surface samples, deg), lsH (level-set mean curvature
error), b999 (worst shape's band error p99.9, sigma), far999 (far-field p99.9, sigma), valL1 / valEik (loss terms on
fixed validation queries).

Noise: GPU training is not deterministic, and seeds change the stream. The seed spread of nbase is the noise band:

| | kerr | slope | aniso | inSad | angAn | angMe | lsH | b999 | far999 | valL1 | valEik |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nbase s1-3 | 0.034-0.041 | 0.73-0.81 | 0.09-0.20 | 0.09-0.60 | 2.74-2.98 | 32.5-33.0 | 0.45-0.48 | 0.82-0.93 | 0.20-0.22 | 0.0036-0.0037 | 0.053-0.054 |

A single-seed result inside the band means "no measurable effect at screening length", not "useless". The screen
runs 9000 steps against L13's 220000: it ranks variants; it does not predict full-length results. The winners have
not been run at full length.

### Always-on changes: base (fa5cead) vs nbase (7b2494e), seeds 1-3

| | kerr | slope | aniso | plane k sigma | sphere k rho | inSad | angAn | angMe | lsH | b999 | far999 | snail p99.9 | valL1 | valEik |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.035-0.045 | 0.74-0.81 | -0.005..0.004 | -0.060..+0.038 | -1.14..-0.98 | 0.003-0.009 | 3.30-3.38 | 31.7-31.8 | 0.51-0.53 | 0.84-0.86 | 0.15-0.19 | 0.680-0.682 | 0.0031-0.0032 | 0.071-0.073 |
| nbase | 0.034-0.041 | 0.73-0.81 | 0.09-0.20 | -0.013..+0.012 | -1.11..-1.06 | 0.09-0.60 | 2.74-2.98 | 32.5-33.0 | 0.45-0.48 | 0.82-0.93 | 0.20-0.22 | 0.670-0.677 | 0.0036-0.0037 | 0.053-0.054 |

The only change of values beyond rounding is the frame form of the blend (§1.1); the effect is attributed to it (not
separated by a run). Better: normal angle at analytic surfaces, level-set curvature, eikonality, anisotropy and
saddles (the old-form models predict no anisotropy at all at this length). Worse: validation L1 distance
(0.0031-0.0032 -> 0.0036-0.0037), far-field p99.9 (0.15-0.19 -> 0.20-0.22 sigma), normal angle at mesh samples
(31.7-31.8 -> 32.5-33.0 deg).

Flat input (eval only, `g-obj/scripts/eval_thin.py`; 2048-point planes and the faces of 1-4 sigma thick plates,
mean curvature H sigma, truth 0): the old-form models carry a seed-dependent offset (released -0.042; base seeds
-0.015, -0.058, +0.024; L13 -0.008), and no point of any old-form model has |k_max sigma| < 1e-2 on a plane. With the
frame form the offset is gone on all three seeds (-0.002 ... -0.006), and 6-13 % of the plane points (phase 2) have
|k_max sigma| < 1e-2. This confirms the review's §1.1 claim that the old form kept flat input curved.

Thin features: the frame form makes the models curve thin features more at screening length.
- Plates 0.15 sigma thick (the opposite face inside every neighbourhood), phase 2: face H sigma -0.18 ... +0.06 for
  base, -0.97 ... -1.05 for nbase (L13 -1.07, released -1.70); wrong-sign interior points 1.0-1.2 % vs 5.9-6.7 %;
  nbase loses the wall on 0.5-1.0 % of the lines through it, base on none (L13 1.5 %). From 0.3 sigma up no model loses
  a wall.
- Renders (mean |true SDF - model SDF| at the model surface, 1e-3 of the shape half-size; 10 CAD shapes, 10 blobs, and
  the snail against an IMLS of its points): CAD 4.7-5.3 (base) -> 3.9-4.5 (nbase), better; blobs 2.09-2.12 both; snail
  0.93-0.97 -> 1.19-1.24, worse: the snail's eye stalks are thinned (stalk pixels thinned by more than 4e-3: 10-12 %
  vs 31-32 %).
- Along the L13 training (old form) the same stalk thinning develops with more steps: stalk mean deviation +5.2e-3 at
  10k steps, -1.6e-3 at the end (220k), 22 % thinned stalk pixels at the end, while CAD, blob and snail errors fall
  overall (not monotonically: CAD 7.2 at 10k, 96 at 20k, 2.7 at 220k). The fixed form reaches at 9k steps what the
  old one reaches after long training; the 0.15-sigma plate result of L13 agrees. So the frame form speeds up
  curvature learning rather than changing the end point. It stays on: it is exact in real arithmetic.
- The hypothesis that the near-zero-curvature cancellation of the old form punished thin sheets was tested: the flat
  input bias is confirmed, the loss of thin walls is not (the opposite happens).

### Flags against nbase

Seeds 1-3 where three ranges are given, else seed 1. Bold: outside the nbase band in the good direction (all seeds),
for the columns where lower is better (not aniso and inSad, ideal 1). s/step: GPU training step time.

| variant | flags | seeds | kerr | aniso | inSad | angAn | angMe | lsH | b999 | far999 | valL1 | valEik | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nbase | | 3 | 0.034-0.041 | 0.09-0.20 | 0.09-0.60 | 2.74-2.98 | 32.5-33.0 | 0.45-0.48 | 0.82-0.93 | 0.20-0.22 | 0.0036-0.0037 | 0.0532-0.0540 | 0.15 |
| perceiver_wls | `--encoder perceiver --wls` | 3 | **0.023-0.028** | 1.15-1.17 | 1.00 | **1.23-1.29** | **27.5-27.9** | **0.19-0.21** | **0.66-0.73** | 0.28-0.31 | **0.0034-0.0035** | 0.0683-0.0687 | 0.06 |
| wsdCE | `--schedule wsd --wsd-decay 0.2 --carry-opt --ema 0.999` | 3 | 0.028-0.038 | 0.41-0.54 | 0.89-0.96 | **1.88-2.10** | **30.8-31.9** | **0.31-0.35** | 0.79-0.91 | 0.21-0.23 | **0.0034-0.0035** | **0.0480-0.0492** | 0.15 |
| nrm | `--normal-w 0.1` | 3 | 0.028-0.048 | 0.30-0.48 | 0.66-0.92 | **1.98-2.52** | **29.9-30.7** | **0.34-0.41** | 0.76-0.83 | 0.18-0.22 | **0.0028-0.0029** | 0.0569-0.0575 | 0.15 |
| mix | `--mix analytic:0.25:...` | 3 | 0.028-0.034 | 0.37-0.46 | 0.80-0.91 | **1.98-2.15** | **30.9-32.2** | **0.33-0.35** | 0.86-0.92 | 0.18-0.21 | 0.0037-0.0039 | 0.0558-0.0572 | 0.16 |
| frame_inv | `--frame invariant` | 3 | 0.025-0.041 | 0.20-0.32 | 0.56-0.77 | 2.55-2.76 | **28.7-29.0** | **0.41-0.43** | 0.76-0.87 | **0.18-0.19** | 0.0037-0.0039 | **0.0480-0.0490** | 0.50 |
| combo | flags of perceiver_wls + wsdCE + nrm + mix | 1 | **0.020** | 1.03 | 1.00 | **0.68** | **28.7** | **0.11** | **0.62** | 0.39 | **0.0031** | 0.0680 | 0.08 |
| aux | `--mix ... --aux-curv 0.05` | 3 | **0.005-0.010** | 0.76-0.81 | 1.00 | 2.57-3.30 | 35.0-37.0 | 0.40-0.54 | 0.88-0.92 | 0.29-0.50 | 0.0033-0.0036 | 0.0897-0.0933 | 0.16 |
| nrmB | `--normal-w 0.1 --normal-band` | 1 | 0.038 | 0.30 | 0.76 | **2.35** | **30.6** | **0.40** | 0.83 | 0.20 | **0.0030** | 0.0565 | |
| corrupt | `--corrupt-flip 0.02 --corrupt-outliers 0.01` | 1 | 0.049 | 0.06 | 0.01 | 3.21 | 33.0 | 0.52 | **0.66** | 0.20 | 0.0038 | **0.0514** | |
| input_mlp | `--input-mlp` | 1 | 0.037 | 0.05 | 0.00 | 3.26 | **31.4** | 0.52 | 1.01 | **0.13** | **0.0034** | **0.0517** | |
| frame_cont | `--frame continuous` | 1 | 0.035 | 0.04 | 0.12 | 3.21 | **31.4** | 0.52 | 0.82 | **0.18** | **0.0032** | 0.0619 | |
| holes | `--holes 0.25:1.5` | 1 | 0.036 | 0.23 | 0.71 | **2.60** | 33.1 | **0.43** | 0.92 | **0.16** | 0.0039 | 0.0534 | |
| distS | `--dist-sigma 0.125` | 1 | 0.040 | 0.24 | 0.37 | **2.67** | 33.1 | **0.44** | 0.90 | 0.20 | 0.0038 | **0.0524** | |
| bandS1 | `--band-sigma 1.0 --min-extent-sigma 0.5` | 1 | 0.040 | 0.21 | 0.67 | **2.70** | 32.5 | **0.45** | 0.86 | 0.20 | 0.0037 | **0.0519** | |
| lip | `--lipschitz-cube` | 1 | 0.038 | 0.15 | 0.41 | 2.82 | 33.1 | 0.47 | 0.91 | 0.20 | 0.0037 | 0.0533 | |
| ema | `--ema 0.999` | 1 | 0.034 | 0.14 | 0.26 | 2.85 | 33.1 | 0.46 | 0.92 | 0.20 | 0.0037 | 0.0536 | |
| carry | `--carry-opt` | 1 | 0.035 | 0.17 | 0.45 | 2.76 | 33.1 | 0.46 | 0.91 | 0.20 | 0.0037 | 0.0532 | |
| facet | `--resample-decimated 2048` | 1 | 0.040 | 0.20 | 0.57 | 2.79 | 32.8 | 0.46 | 0.88 | 0.20 | 0.0037 | **0.0526** | |
| facetA | `--mix ... --resample-decimated 2048 --resample-analytic 2048:32768` (compare with mix) | 1 | 0.039 | 0.43 | 0.89 | 1.98 | 32.2 | 0.33 | 0.90 | 0.19 | 0.0038 | 0.0560 | |
| seedA | `--seed-area` | 1 | 0.037 | 0.15 | 0.37 | 2.82 | 33.1 | 0.48 | 0.89 | 0.20 | 0.0037 | **0.0529** | |
| nnoise3 | `--normal-noise 3` | 1 | 0.036 | 0.19 | 0.74 | 2.87 | 32.6 | 0.49 | 0.84 | 0.20 | 0.0038 | **0.0525** | |
| halo | `--union-k 96` | 1 | 0.057 | 0.09 | 0.66 | 3.00 | 33.5 | 0.49 | 0.88 | 0.21 | **0.0034** | **0.0517** | 0.15 |
| perceiver | `--encoder perceiver` | 1 | 0.060 | 0.06 | 0.00 | 3.11 | 33.8 | 0.49 | 0.84 | 0.22 | 0.0037 | 0.0564 | 0.05 |
| auxheads | `--aux-heads --corrupt-flip 0.02 --corrupt-outliers 0.01` | 1 | 0.096 | 0.01 | 0.00 | 32.8 | 51.4 | 0.82 | 3.96 | 11.5 | 0.0132 | 0.3327 | 0.15 |

Out of memory at 6 shapes on the 16 GB GPU (allocations of 10.8 GiB for `--wls`, 12.0 GiB for `--patch-c 64 --upad
320`); measured with fewer shapes against their own reference:

| variant | flags | kerr | aniso | inSad | angAn | angMe | lsH | b999 | far999 | valL1 | valEik |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nbase_ms4 | `--multishape 4` | 0.037 | 0.14 | 0.39 | 2.92 | 32.9 | 0.49 | 0.91 | 0.19 | 0.0038 | 0.055 |
| wls_ms4 | `--wls --multishape 4` | 0.044 | 1.24 | 1.00 | 2.27 | 29.6 | 0.35 | 0.79 | 0.29 | 0.0040 | 0.074 |
| wls0_ms4 | `--wls --wls-bound 0 --multishape 4` | 0.036 | 1.20 | 1.00 | 0.53 | 28.5 | 0.09 | 1.08 | 1.81 | 0.0059 | 0.127 |
| patchC_ms3 | `--patch-c 64 --upad 320 --multishape 3` (192 centres per step, as 6 x 32) | 0.045 | 0.06 | 0.24 | 3.04 | 33.1 | 0.49 | 0.90 | 0.20 | 0.0036 | 0.055 |

Reading:
- Confirmed gains (3 seeds): `--encoder perceiver --wls` (best surface and curvature numbers and band tail, 2.5x faster
  steps; costs: far-field p99.9 0.28-0.31 vs 0.20-0.22 sigma, eikonality 0.068, anisotropy slope overshoots to 1.15);
  `--schedule wsd --wsd-decay 0.2 --carry-opt --ema 0.999` (surface and curvature, best eikonality, no cost seen);
  `--normal-w 0.1` (best validation L1, mesh normal angle 29.9-30.7 vs 32.5-33.0 deg; eikonality 0.057 slightly
  worse); `--mix analytic:0.25` (surface and curvature; validation L1 and eikonality slightly worse);
  `--frame invariant` (mesh normal angle 28.7-29.0 deg, eikonality, lsH, far field; small angAn gain; 3.3x step time).
- The gains stack: the four cheap ones together (seed 1) give angAn 0.68 deg and lsH 0.11, against 2.74-2.98 and
  0.45-0.48; the far-field cost of the WLS stacks too (0.39 sigma).
- `--wls` alone helps surface and curvature (angAn 2.92 -> 2.27, lsH 0.49 -> 0.35 vs its 4-shape reference) and costs
  far field and eikonality (0.19 -> 0.29 sigma, 0.055 -> 0.074). Pure WLS (`--wls-bound 0`) is nearly exact on the
  analytic shapes (angAn 0.53 deg) and breaks the field away from the surface (far p99.9 1.81 sigma, validation L1
  0.0059): the learned residual is needed.
- `--aux-curv`: the curvature metrics become nearly perfect (kerr 0.005-0.010, slope 0.97-1.01, inSad 1.0) but the
  field gets worse (eikonality ~0.09, far field 0.29-0.50, mesh normal angle 35-37 deg).
- `--corrupt-flip 0.02 --corrupt-outliers 0.01`: on clean data a better band tail, the curvature columns slightly
  worse; on corrupted input a large robustness gain (next section). `--input-mlp` and `--frame continuous` (one seed): far field and mesh angle
  better, surface curvature and anisotropy worse (their curvature columns look like the old-form base).
  `--normal-w 0.1 --normal-band` is a smaller gain than `--normal-w 0.1` alone.
- No measurable effect at this length (single seed, inside or at the edge of the band): `--lipschitz-cube`,
  `--ema` alone, `--carry-opt` alone, `--resample-decimated` (also with `--resample-analytic`, against `--mix`),
  `--seed-area`, `--normal-noise 3`, `--patch-c 64 --upad 320` (at 3 shapes; slightly worse than the 4-shape
  reference, confounded with fewer shapes per step). Small single-seed differences just outside the band, not
  repeated: `--holes 0.25:1.5` (angAn 2.60, lsH 0.43, far 0.16), `--dist-sigma 0.125` (angAn 2.67, lsH 0.44),
  `--band-sigma 1.0 --min-extent-sigma 0.5` (angAn 2.70, lsH 0.45); a three-seed band understates the spread, so
  these are candidates at most.
- Worse: `--union-k 96` (kerr 0.057), `--encoder perceiver` alone (kerr 0.060, no saddles).
- Fails: `--aux-heads` (with corruption). The loss falls to 0.031 at step 750 and then rises to 0.06-0.07, where it
  stays (`--corrupt-*` alone: 0.019 at step 3000); the field is broken (analytic normal angle 33 deg; 14-18 % of the
  band samples have the wrong sign, on clean and corrupted input). The flag is kept but does not work.
- Gate: every MISS of the screening is the obsolete plane check (above), or the far-field check of a phase-1 model
  (corrupt 1.16, wls_ms4 1.02, aux seed 1 0.63 sigma; all pass at phase 2), or a broken variant (auxheads, wls0).

### Zero-shot frame averaging, corrupted input, forward cost

`eval_model.py --frame invariant` on the nbase phase-2 weights (trained in the authors' frame, no retraining):

| | kerr | aniso | inSad | angAn | angMe | lsH | b999 | far999 | valL1 | valEik |
|---|---|---|---|---|---|---|---|---|---|---|
| nbase s1-3, own frame | 0.034-0.041 | 0.09-0.20 | 0.09-0.60 | 2.74-2.98 | 32.5-33.0 | 0.45-0.48 | 0.82-0.93 | 0.20-0.22 | 0.0036-0.0037 | 0.053-0.054 |
| nbase s1-3, `--frame invariant` | 0.035-0.039 | 0.05-0.13 | 0.01-0.36 | 2.88-2.92 | 31.8-32.4 | 0.46-0.47 | 0.83-0.94 | 0.19-0.21 | 0.0036-0.0037 | 0.052-0.054 |

No measurable change at screening length except a slightly lower mesh normal angle (31.8-32.4 vs 32.5-33.0 deg)
and fewer saddles. On the L13
weights (CPU) the same averaging improved every analytic metric (angAn 1.10 -> 0.89 deg, lsH 0.177 -> 0.155), so it
helps a well-trained model; training with `--frame invariant` (above) gains more.

Corrupted input (`g-arch/scripts/eval_corrupt.py`, phase-2 models of seed 1): the 32-NN field of clean and corrupted
clouds (2 % of the normals flipped, 1 % of the points moved off the surface as outliers, or both) against the clean
ground truth; worst shape's band error p99 in sigma / share of band samples with the wrong sign:

| model | clean | 2 % flips | 1 % outliers | both |
|---|---|---|---|---|
| base (before the review) | 0.565 / 0.27 % | 1.270 / 0.78 % | 2.690 / 2.48 % | 2.007 / 3.29 % |
| nbase | 0.578 / 0.31 % | 2.141 / 1.95 % | 2.450 / 1.78 % | 2.041 / 2.84 % |
| corrupt (`--corrupt-flip 0.02 --corrupt-outliers 0.01`) | **0.390 / 0.10 %** | **0.429 / 0.16 %** | **1.502 / 0.31 %** | **1.281 / 0.67 %** |
| auxheads (diverged run) | 3.745 / 17.8 % | 3.578 / 14.7 % | 3.583 / 16.2 % | 3.452 / 14.3 % |

Training on corrupted input makes the field nearly insensitive to isolated flipped normals (wrong sign 1.95 % ->
0.16 %) and much less sensitive to outliers, and its clean tail is better too (single seed). The review's §3.3 claim
that a six-coefficient torus cannot undo a flipped normal holds for the flipped point's own torus, but the blended
field of a network trained with corruption stays correct without the extra heads (how, e.g. through the neighbours'
tori, was not analysed). Whether `--aux-heads` helps on top needs a run that does not diverge.

GPU forward cost of the architecture variants (`g-arch/scripts/bench_arch.py`, batch 4096 neighbourhoods; 16384 and
50000 do not fit next to the desktop; per 1000 neighbourhoods of k = 64):

| variant | params | GFLOP | ms | time vs default |
|---|---|---|---|---|
| default | 1,588,358 | 200.0 | 25.0 | 1 |
| `--frame continuous` | 1,588,358 | 200.0 | 25.1 | 1.01 |
| `--frame invariant` | 1,588,358 | 424.2 | 75.5 | 3.02 |
| `--input-mlp` | 1,605,766 | 200.2 | 25.4 | 1.02 |
| `--encoder perceiver` | 1,226,630 | 29.8 | 3.6 | 0.14 |
| `--wls` | 1,588,616 | 223.6 | 28.3 | 1.13 |
| `--encoder perceiver --wls` | 1,425,416 | 52.1 | 5.95 | 0.24 |
| `--aux-heads` | 1,588,616 | 200.0 | 25.2 | 1.01 |

The default's 200 GFLOP include the token-0 readout (224 GFLOP before). `--encoder perceiver --wls`, the best variant
of the screen, precomputes the coefficients 4.2x faster than the default network.

### Not implemented

- Analytic scenes with creases and CSG (§4.1): the exact 2D primitives (`shape_3d.SDF3D`) evaluate through the C++
  bindings, not JAX; a port of the primitives, extrusion / revolution, CSG and exact surface sampling with
  curvatures is about a day of work.
- An O(2)-steerable network (§3.1): needs new attention, normalization and tensor-product layers; frame averaging
  (`--frame invariant`) gives exact invariance with the existing network and weights at 3x network cost.
- A cancellation-free eigenvector formula for the case v2 = 0 exactly (§1.2; also in C++): never seen on real data.
- The (theta, phi) round trip of the axis in the C++ evaluator (§1.2): in double it costs up to ~1e-4 sigma for the
  flattest L13 tori, negligible for the released weights; unchanged so that released results stay identical.
- A 128-point subsample for the noise sigma (§4.4; it changes values), parallel prefetch workers (§4.4; the prefetch
  of 45-64 ms is hidden behind the GPU step), a best-checkpoint file (§4.2; a feature, not a fix).
