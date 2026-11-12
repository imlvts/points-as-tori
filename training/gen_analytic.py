"""Generate analytic training scenes (exact SDF, normals and principal curvatures) for train.py --mix.

	python gen_analytic.py OUT.npz N SEED [workers]

Families (shares in WEIGHTS): closed tubes (radius log-uniform 0.1-1.0 sigma), sticks-only scenes, thin walls of
revolution and bowls (half-thickness log-uniform 0.15-1.0 sigma), tori (inner halves are saddles), spheres, rounded
boxes, mixed; sigma is the median 64-NN distance of the 2048-point sample. Each scene is a union of pairwise disjoint
primitives inside [-0.9, 0.9]^3, so the min of the primitive SDFs is the exact SDF.
Per scene: prims (8, 16), types (8), polys (3, 512, 3), pcount (3) (the encoding of analytic_sdf.py), the 2048-point
sample P, N, k1, k2, ptype (point class), pid (primitive), sigma, family, nprim.
Our runs used N = 6000, SEED = 1 for training (about 1 min with 16 workers, 284 MB)."""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import analytic_scene as AS  # noqa: E402

FAMILIES = ['tubes', 'sticks', 'walls', 'tori', 'spheres', 'boxes', 'mixed']
WEIGHTS = [0.22, 0.13, 0.30, 0.12, 0.07, 0.08, 0.08]
TUBE_R = (0.1, 1.0)
WALL_T = (0.15, 1.0)


def one(args):
	i, seed = args
	rng = np.random.default_rng([seed, i, 13])
	fam = rng.choice(FAMILIES, p=WEIGHTS)
	prims, meta = AS.make_scene(fam, rng, tube_r=TUBE_R, wall_t=WALL_T)
	P, N, k1, k2, pt, pid = AS.sample_scene(prims, 2048, rng)
	sig = AS.sigma_of(P)
	E, T, polys, pc = AS.encode(prims)
	return dict(prims=E, types=T, polys=polys, pcount=pc, P=P, N=N, k1=k1, k2=k2, ptype=pt, pid=pid, sigma=np.float32(sig),
				family=fam, nprim=len(prims))


if __name__ == '__main__':
	out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
	workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
	t0 = time.time()
	with Pool(workers) as pool:
		rows = pool.map(one, [(i, seed) for i in range(n)], chunksize=8)
	keys = ['prims', 'types', 'polys', 'pcount', 'P', 'N', 'k1', 'k2', 'ptype', 'pid', 'sigma']
	d = {k: np.stack([r[k] for r in rows]) for k in keys}
	d['family'] = np.array([r['family'] for r in rows])
	d['nprim'] = np.array([r['nprim'] for r in rows])
	np.savez_compressed(out, **d)
	fam, cnt = np.unique(d['family'], return_counts=True)
	pt, ks = d['ptype'], np.abs(d['k1']) * d['sigma'][:, None]
	print(f'{out}: {n} shapes in {time.time() - t0:.0f} s; families', dict(zip(fam.tolist(), cnt.tolist())),
		  'sigma median', float(np.median(d['sigma'])))
	print(f'points: tube side {np.mean(pt == 1) * 100:.1f}% (|k1|s>=2: {np.mean((pt == 1) & (ks >= 2)) * 100:.1f}%, >=4: '
		  f'{np.mean((pt == 1) & (ks >= 4)) * 100:.1f}%), wall side {np.mean(pt == 7) * 100:.1f}%, wall rim {np.mean(pt == 8) * 100:.1f}%, '
		  f'saddle {np.mean(pt == 3) * 100:.1f}%, box face {np.mean(pt == 4) * 100:.1f}%')
