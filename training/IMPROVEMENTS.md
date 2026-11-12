# Training improvements

Changes to the training code of this repository (MIT, (c) 2026 Nicole Feng), from a study of the trainer in
November 2026. Finding ids (F*, X*) refer to that study's notes. The architecture (`FundamentalFormPredictor`) is
unchanged.

## Bug fixes (always on)

| fix | finding | effect |
|---|---|---|
| Guard-free log-sum-exp form of the Eq. 25 blend (`network.py`, `torus_distance_gradient_masked`) | F2 | The blend took sigma/lambda from the 64 nearest points but summed over all points and divided by `max(sum W, 1e-10)`; for queries beyond ~0.9x the 64-NN radius every weight underflowed and the trainer saw phi ~ 0 with a meaningless gradient, which rewards over-curved tori. Same value wherever the guard did not fire. |
| Geometric sign rule: torus center from kappa_min (`torus_precompute_masked` and the C++ `fit_torus_at_point`) | F4 | At saddles the center was put on the wrong side (the mirror image of the osculating torus, dg/dn* = -1 at the point), a loss cliff at kappa_min kappa_max = 0. Non-saddle tori are bit-identical; with the released weights 1 of 4096 snail tori and 35 of 12917 holey-bunny tori change. Training and inference use the same rule. |
| Padded shapes give finite, correct gradients (`InputFeatures`, `torus_precompute_masked`, loader, train loop) | F3 | Padded rows (neighbors = -1) had a zero neighborhood scale, 0/0 = NaN features and an all-NaN gradient that was zeroed: about 45 % of the simulated ABC scans gave no training signal. Now the gradient of a padded shape equals that of the same shape without padding (rel. 3.5e-5); batches of padding only are skipped. |
| Local zero-set / narrow-band query sampler (`mesh_sampling_jax.py`) | X9a | Zero-set and band queries were sampled on whole faces whose bounding box meets the neighborhood box; on CAD meshes 32-87 % of them were outside the neighborhood. Now rejection sampling on mesh ∩ box (0 % outside). |
| Unit normals on load (`load_mesh_training_data`) | F1 | `pretraining_blob.npz` and the validation files store normals of length 2, which made over-curvature free in pretraining and the validation numbers meaningless. |

Checked and left unchanged: the phase-1 learning rate is 1e-5 in the code but 1e-6 in the paper (Sec. 4.3); it is
1e-5 in every commit since the first one, from which the released weights come.

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
| `--gate` | | evaluate every phase's model with `eval_model.py` and print the gate checks (log only) |
| `--preset l13` | | our best recipe (L13) in one command; explicit flags override it |

`--resample`, `--noise`, `--quantize` and `--mix` need `--multishape` (the authors' loop generates queries for every
point of a shape and uses the stored neighbors).

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

Check (1500 steps of this recipe on blobs + ABC17 with 100 warmup steps, seed 1, against our original trainer's
equivalent runs; the patch stream is the same, so the per-step losses nearly coincide: final loss 0.01548 vs 0.01551):

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
