"""
Held-out quality check of a trained fundamental-form predictor (about 2 minutes on one GPU), and the phase gate.

	python eval_model.py MODEL [MODEL ...] [--out DIR] [--data-dir DATASETS] [--gate] [--no-loss]

MODEL: a model saved by train.py (.pkl), or [LABEL=]PATH. Writes DIR/<label>.json and prints one summary row per model.
All errors are in units of sigma, the median distance to the 64 nearest neighbors (the network's own scale).

1. Curvature tracking on analytic clouds (sphere, cylinder, torus, plane at several densities; fixed seeds): the
   network's principal curvatures vs the truth.
   kerr: mean |H_pred - H_true| sigma over the clouds (and the outer / inner torus regions), H the mean curvature;
   track_slope / track_icpt / track_r: least-squares fit of predicted vs true H sigma (ideal 1 / 0 / 1);
   aniso_slope: the same for (|k1 - k2| / 2) sigma (ideal 1); plane_ks: k_max sigma on the plane (truth 0);
   sph_kr: k_max rho on the 2048-point sphere of radius rho (truth -1); inner_saddle: fraction of torus inner-ring
   points predicted as saddles (truth 1).
2. Surface: the blended field at fresh surface samples (not the input points) of analytic shapes and of three meshes
   (data/block, femur, knot; 512 points each): angle between grad phi and the true normal, |phi|, and on analytic
   shapes the mean curvature of the level set (lsH_err).
3. Error tails of the blended field: per ground-truth shape (sphere, torus, plane, cylinder, block, femur, knot) the
   band error |phi - SDF| at surface samples offset by U(-2, 2) sigma along the normal, and the far-field error at
   points of the doubled bounding box with |SDF| > 4 sigma; on the snail (no ground truth) outward probes
   p + delta sigma_p n_p (delta = 0.5, 1). Headlines: the worst shape's p99.9.
4. Loss (unless --no-loss): the training loss terms (L1 distance, eikonality) on fixed queries (256 fixed rows per
   shape) of the validation file (DATA_DIR/ABC17_validation_3.npz, unit normals), all-points and 32-NN blend.

The field is the training blend of network.py (geometric sign rule, guard-free log-sum-exp weights) over all points.

Gate (logged, never aborts): kerr <= 0.1; analytic normal angle <= 4 deg; sphere k rho within 30 % of -1;
plane k sigma < 0; snail outward p99.9 < 1 sigma; far-field p99.9 < 0.5 sigma (worst ground-truth shape).
"""

import argparse
import json
import os
import re
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.spatial import cKDTree

from pointsastori.network import FundamentalFormPredictor, torus_precompute_masked, torus_distance_gradient_masked
import mesh_sampling_jax as MS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA = os.path.join(REPO, 'data')
MESHES = ['block', 'femur', 'knot']


# ---------------------------------------------------------------- models
def load_model(path):
	"""-> (model in eval mode, k_neighbors). A train.py .pkl, or an .npz of flat Flax paths with a '__config__' entry."""
	if path.endswith('.npz'):
		z = np.load(path)
		cfg = json.loads(str(z['__config__']))
		model = FundamentalFormPredictor(
			embed_dim=cfg['embed_dim'],
			n_freq=cfg.get('n_freq', 0) or 0,
			n_layers=cfg['n_layers'],
			n_heads=cfg['n_heads'],
			mlp_dim=cfg['mlp_dim'],
			dropout_rate=0.0,
			rngs=nnx.Rngs(0),
		)
		state = nnx.state(model, nnx.Param)
		leaves, treedef = jax.tree_util.tree_flatten_with_path(state)

		def key_of(p):  # flat Flax path, e.g. 'blocks.0.attention.query.kernel'
			key = re.sub(r"[\[\]']+", '.', jax.tree_util.keystr(p)).strip('.')
			return re.sub(r'\.(value|raw_value)$', '', re.sub(r'\.+', '.', key)).strip('.')

		new = [jnp.asarray(z[key_of(p)], x.dtype) for p, x in leaves]
		nnx.update(model, jax.tree_util.tree_unflatten(treedef, new))
		nnx.eval_mode(model)
		return model, cfg['k_neighbors']
	return FundamentalFormPredictor.load_saved_model(path)


def coeffs(model, P, N, nbr, chunk=1024):
	out = []
	for s in range(0, len(nbr), chunk):
		idx = nbr[s : s + chunk]
		out.append(np.asarray(model(jnp.asarray(P[idx], jnp.float32), jnp.asarray(N[idx], jnp.float32))))
	return np.concatenate(out)


def curvatures(c):
	"""Principal curvatures of the predicted quadric (convex with outward normal < 0), sorted |k_max| >= |k_min|."""
	c = c.astype(np.float64)
	a00, a01, a10, a11, a02, a20 = c.T
	A = np.sqrt(1 + a01**2 + a10**2)
	I = np.stack([np.stack([1 + a10**2, a10 * a01], -1), np.stack([a10 * a01, 1 + a01**2], -1)], -2)
	II = np.stack([np.stack([2 * a20, a11], -1), np.stack([a11, 2 * a02], -1)], -2) / A[:, None, None]
	ev = np.linalg.eigvals(np.linalg.solve(I, II)).real
	big = np.abs(ev[:, 0]) >= np.abs(ev[:, 1])
	return np.where(big, ev[:, 0], ev[:, 1]), np.where(big, ev[:, 1], ev[:, 0])


def field(P, N, C, knn=0):
	"""Blended SDF (value, gradient) of the tori of cloud P, N with coefficients C."""
	mask = jnp.ones(len(P), bool)
	Pj = jnp.asarray(P, jnp.float32)
	cen, ax, R, r = torus_precompute_masked(mask, Pj, jnp.asarray(N, jnp.float32), jnp.asarray(C, jnp.float32))

	def f(Q, chunk=512):
		ds, gs = [], []
		for s in range(0, len(Q), chunk):
			q = np.zeros((chunk, 3), np.float32)
			m = min(chunk, len(Q) - s)
			q[:m] = Q[s : s + m]
			d, g = torus_distance_gradient_masked(mask, jnp.asarray(q)[None], Pj, cen, ax, R, r, knn=knn)
			ds.append(np.asarray(d[0])[:m])
			gs.append(np.asarray(g[0])[:m])
		return np.concatenate(ds).astype(np.float64), np.concatenate(gs).astype(np.float64)

	return f


# ---------------------------------------------------------------- analytic clouds (fixed seeds)
def _analytic_clouds():
	rng = np.random.default_rng(0)

	def sphere(n, rho=0.8):
		x = rng.normal(size=(n, 3))
		x /= np.linalg.norm(x, axis=1, keepdims=True)
		return rho * x, x.copy(), np.tile([-1 / rho, -1 / rho], (n, 1)), np.ones(n, bool)

	def cylinder(n, rho=0.4, h=1.6):
		th, z = rng.uniform(0, 2 * np.pi, n), rng.uniform(-h / 2, h / 2, n)
		N = np.stack([np.cos(th), np.sin(th), 0 * th], 1)
		return np.c_[rho * N[:, :2], z], N, np.tile([-1 / rho, 0.0], (n, 1)), np.abs(z) < 0.4

	def torus(n, R=0.6, r=0.25):
		out = []
		while sum(len(o) for o in out) < n:
			u, v = rng.uniform(0, 2 * np.pi, 4 * n), rng.uniform(0, 2 * np.pi, 4 * n)
			keep = rng.uniform(0, 1, 4 * n) < (R + r * np.cos(v)) / (R + r)
			out.append(np.c_[u[keep], v[keep]])
		u, v = np.concatenate(out)[:n].T
		N = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], 1)
		P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
		k2 = -np.cos(v) / (R + r * np.cos(v))
		return P, N, np.c_[np.full(n, -1 / r), k2], np.ones(n, bool)

	def plane(n, L=1.8):
		P = np.c_[rng.uniform(-L / 2, L / 2, (n, 2)), np.zeros(n)]
		return P, np.tile([0.0, 0, 1], (n, 1)), np.zeros((n, 2)), np.abs(P[:, :2]).max(1) < 0.45

	S = {}
	for n in (256, 512, 1024, 2048, 8192):
		S[f'sphere{n}'] = sphere(n)
	for n in (512, 2048, 8192):
		S[f'cyl{n}'] = cylinder(n)
	for n in (1024, 4096, 16384):
		S[f'torus{n}'] = torus(n)
	for n in (512, 2048, 8192):
		S[f'plane{n}'] = plane(n)
	return S


SHAPES = _analytic_clouds()


def knn_of(P, k):
	return cKDTree(P).query(P, k=k)[1].astype(np.int32)


def analytic_fresh(name, n, rng):
	"""Fresh surface samples, true normals and true mean curvature of an analytic shape."""
	if name.startswith('sphere'):
		x = rng.normal(size=(n, 3))
		x /= np.linalg.norm(x, axis=1, keepdims=True)
		return 0.8 * x, x, np.full(n, -1 / 0.8)
	if name.startswith('cyl'):
		th, z = rng.uniform(0, 2 * np.pi, n), rng.uniform(-0.4, 0.4, n)
		N = np.stack([np.cos(th), np.sin(th), 0 * th], 1)
		return np.c_[0.4 * N[:, :2], z], N, np.full(n, -0.5 / 0.4)
	if name.startswith('torus'):
		R, r = 0.6, 0.25
		out = []
		while sum(len(o) for o in out) < n:
			u, v = rng.uniform(0, 2 * np.pi, 4 * n), rng.uniform(0, 2 * np.pi, 4 * n)
			keep = rng.uniform(0, 1, 4 * n) < (R + r * np.cos(v)) / (R + r)
			out.append(np.c_[u[keep], v[keep]])
		u, v = np.concatenate(out)[:n].T
		N = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], 1)
		P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
		return P, N, 0.5 * (-1 / r - np.cos(v) / (R + r * np.cos(v)))
	if name.startswith('plane'):
		P = np.c_[rng.uniform(-0.45, 0.45, (n, 2)), np.zeros(n)]
		return P, np.tile([0.0, 0, 1], (n, 1)), np.zeros(n)
	raise ValueError(name)


def analytic_sdf(name, Q):
	if name.startswith('sphere'):
		return np.linalg.norm(Q, axis=1) - 0.8
	if name.startswith('torus'):
		return np.hypot(np.linalg.norm(Q[:, :2], axis=1) - 0.6, Q[:, 2]) - 0.25
	if name.startswith('plane'):
		return Q[:, 2].copy()
	if name.startswith('cyl'):
		return np.linalg.norm(Q[:, :2], axis=1) - 0.4
	raise ValueError(name)


# ---------------------------------------------------------------- meshes of the repository's sample data
def read_obj(path):
	V, F = [], []
	for line in open(path):
		t = line.split()
		if not t:
			continue
		if t[0] == 'v':
			V.append([float(x) for x in t[1:4]])
		elif t[0] == 'f':
			F.append([int(x.split('/')[0]) - 1 for x in t[1:4]])
	V, F = np.array(V), np.array(F, np.int64)
	c = 0.5 * (V.min(0) + V.max(0))
	V = (V - c) / np.abs(V - c).max()  # bounding box centered, max |coordinate| 1
	return V, F


def mesh_fresh(V, F, n, rng):
	a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
	f = rng.choice(len(F), n, p=a / a.sum())
	u, v = rng.uniform(size=n), rng.uniform(size=n)
	flip = u + v > 1
	u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
	P = V[F[f, 0]] + u[:, None] * (V[F[f, 1]] - V[F[f, 0]]) + v[:, None] * (V[F[f, 2]] - V[F[f, 0]])
	Nf = np.cross(V[F[f, 1]] - V[F[f, 0]], V[F[f, 2]] - V[F[f, 0]])
	return P, Nf / np.linalg.norm(Nf, axis=1, keepdims=True)


def mesh_sdf(V, F, Q, chunk=2048):
	Vj, Fj, FM = jnp.asarray(V, jnp.float32), jnp.asarray(F, jnp.int32), jnp.ones(len(F), bool)
	f = jax.jit(lambda q: MS.signed_distance_to_mesh(q, Vj, Fj, FM))
	out = []
	for s in range(0, len(Q), chunk):
		q = np.zeros((chunk, 3), np.float32)
		m = min(chunk, len(Q) - s)
		q[:m] = Q[s : s + m]
		out.append(np.asarray(f(jnp.asarray(q)))[:m])
	return np.concatenate(out).astype(np.float64)


_MESH = {}


def mesh_case(name):
	"""(V, F, P, N): the normalized mesh and a fixed 512-point sample of it."""
	if name not in _MESH:
		V, F = read_obj(os.path.join(SAMPLE_DATA, name + '.obj'))
		P, N = mesh_fresh(V, F, 512, np.random.default_rng(100 + MESHES.index(name)))
		_MESH[name] = (V, F, P.astype(np.float32), N.astype(np.float32))
	return _MESH[name]


def load_snail():
	raw = open(os.path.join(SAMPLE_DATA, 'snail.ply'), 'rb').read()
	h = raw.index(b'end_header\n') + len(b'end_header\n')
	a = np.frombuffer(raw[h:], dtype='<f4').reshape(-1, 6)
	P, N = a[:, :3].astype(np.float32), a[:, 3:].astype(np.float32)
	return P, N / np.linalg.norm(N, axis=1, keepdims=True)


# ---------------------------------------------------------------- 1. curvature tracking
def curvature_tracking(model, k):
	rows, pairs = {}, []
	for name, (P, N, K, keep) in SHAPES.items():
		nb = knn_of(P, k)
		C = coeffs(model, P.astype(np.float32), N.astype(np.float32), nb)
		sig = np.median(np.linalg.norm(P[nb] - P[:, None], axis=-1), axis=1)
		kmax, kmin = curvatures(C)
		Kt = np.take_along_axis(K, np.argsort(-np.abs(K), axis=1), axis=1)
		ks = lambda a: float(np.median((a * sig)[keep]))  # noqa: E731
		row = dict(kmax_s=ks(kmax), kmin_s=ks(kmin), tmax_s=ks(Kt[:, 0]), tmin_s=ks(Kt[:, 1]), sigma=float(np.median(sig[keep])))
		if name.startswith('torus'):
			v = np.arctan2(P[:, 2], np.linalg.norm(P[:, :2], axis=1) - 0.6)
			for reg, sel in (('outer', np.cos(v) > 0.9), ('inner', np.cos(v) < -0.9)):
				for a, b in (('kmax', kmax), ('kmin', kmin)):
					row[f'{reg}_{a}_s'] = float(np.median((b * sig)[sel]))
				row[f'{reg}_tmax_s'] = float(np.median((Kt[:, 0] * sig)[sel]))
				row[f'{reg}_tmin_s'] = float(np.median((Kt[:, 1] * sig)[sel]))
				pairs += [(row[f'{reg}_tmax_s'], row[f'{reg}_kmax_s']), (row[f'{reg}_tmin_s'], row[f'{reg}_kmin_s'])]
			row['inner_saddle'] = float(np.mean((kmax * kmin)[np.cos(v) < -0.9] < 0))
		else:
			pairs += [(row['tmax_s'], row['kmax_s']), (row['tmin_s'], row['kmin_s'])]
		rows[name] = row
	t, p = np.array(pairs).T
	th, ph = (t[0::2] + t[1::2]) / 2, (p[0::2] + p[1::2]) / 2
	ta, pa = np.abs(t[0::2] - t[1::2]) / 2, np.abs(p[0::2] - p[1::2]) / 2
	slope, icpt = np.polyfit(th, ph, 1)
	summ = dict(
		track_slope=float(slope),
		track_icpt=float(icpt),
		track_r=float(np.corrcoef(th, ph)[0, 1]),
		kerr=float(np.mean(np.abs(ph - th))),
		aniso_slope=float(np.polyfit(ta, pa, 1)[0]),
		plane_ks=rows['plane2048']['kmax_s'],
		sph_kr=rows['sphere2048']['kmax_s'] / rows['sphere2048']['sigma'] * 0.8,
		sph8k_kr=rows['sphere8192']['kmax_s'] / rows['sphere8192']['sigma'] * 0.8,
		inner_saddle=rows['torus4096']['inner_saddle'],
	)
	return summ, rows


# ---------------------------------------------------------------- 2. surface metrics
def surface_metrics(model, k, n=500):
	rng = np.random.default_rng(11)
	rows = {}
	analytic = ('sphere2048', 'sphere8192', 'cyl2048', 'torus4096', 'plane2048')
	for name in list(analytic) + MESHES:
		if name in analytic:
			P, N = SHAPES[name][0], SHAPES[name][1]
			Q, Nt, Ht = analytic_fresh(name, 3 * n, rng)
		else:
			V, F, P, N = mesh_case(name)
			Q, Nt = mesh_fresh(V, F, 3 * n, rng)
			Ht = None
		nb = knn_of(P, k)
		sig_i = np.median(np.linalg.norm(P[nb] - P[:, None], axis=-1), axis=1)
		dist, near = cKDTree(P).query(Q)
		keep = dist > 0.25 * sig_i[near]  # each torus passes through its own point
		Q, Nt, sq = Q[keep][:n], Nt[keep][:n], sig_i[near][keep][:n]
		Ht = Ht[keep][:n] if Ht is not None else None
		C = coeffs(model, P.astype(np.float32), N.astype(np.float32), nb)
		f = field(P, N, C)
		phi, g = f(Q)
		ang = np.degrees(np.arccos(np.clip(np.sum(g * Nt, 1) / np.maximum(np.linalg.norm(g, axis=1), 1e-12), -1, 1)))
		row = dict(phi_s=float(np.nanmean(np.abs(phi) / sq)), ang=float(np.nanmean(ang)), ang_p95=float(np.nanpercentile(ang, 95)))
		if Ht is not None:
			h = 0.05 * np.median(sq)
			div = np.zeros(len(Q))
			for axis in range(3):
				e = np.zeros(3)
				e[axis] = h
				_, gp = f(Q + e)
				_, gm = f(Q - e)
				div += (gp[:, axis] / np.linalg.norm(gp, axis=1) - gm[:, axis] / np.linalg.norm(gm, axis=1)) / (2 * h)
			row['lsH_err'] = float(np.nanmean(np.abs(-0.5 * div - Ht) * sq))
		rows[name] = row
	summ = {}
	for grp, names in (('an', analytic), ('mesh', MESHES)):
		for key in ('phi_s', 'ang', 'ang_p95', 'lsH_err'):
			vals = [rows[nm][key] for nm in names if key in rows[nm]]
			if vals:
				summ[f'surf_{grp}_{key}'] = float(np.mean(vals))
	return summ, rows


# ---------------------------------------------------------------- 3. error tails
def _q(e, qs=(50, 90, 99, 99.9, 100)):
	e = e[np.isfinite(e)]
	return [float(x) for x in np.percentile(e, qs)] if len(e) else [float('nan')] * len(qs)


def error_distributions(model, k):
	rng = np.random.default_rng(21)
	rows = {}
	for name in ['sphere2048', 'torus4096', 'plane2048', 'cyl2048'] + MESHES:
		if name in MESHES:
			V, F, P, N = mesh_case(name)
			S, NS = mesh_fresh(V, F, 50000, rng)
			sdf = lambda Q: mesh_sdf(V, F, Q)  # noqa: E731
		else:
			P, N = SHAPES[name][0].astype(np.float32), SHAPES[name][1].astype(np.float32)
			S, NS, _ = analytic_fresh(name, 50000, rng)
			sdf = lambda Q: analytic_sdf(name, Q)  # noqa: E731
		nb = knn_of(P, k)
		sig = float(np.median(np.linalg.norm(P[nb] - P[:, None], axis=-1)))
		B = S + (rng.uniform(-2, 2, len(S)) * sig)[:, None] * NS
		f = field(P, N, coeffs(model, P, N, nb))
		row = {'band_abs': _q(np.abs(f(B)[0] - sdf(B)) / sig)}
		if not (name.startswith('plane') or name.startswith('cyl')):  # closed shapes only
			c, h = 0.5 * (P.min(0) + P.max(0)), P.max(0) - P.min(0)
			Fq = c + rng.uniform(-1, 1, (80000, 3)) * h
			DF = sdf(Fq)
			keep = np.abs(DF) > 4 * sig
			Fq, DF = Fq[keep][:20000], DF[keep][:20000]
			row['far_abs'] = _q(np.abs(f(Fq)[0] - DF) / sig)
		rows[name] = row
	P, N = load_snail()
	nb = knn_of(P, k)
	sig_p = np.median(np.linalg.norm(P[nb] - P[:, None], axis=-1), axis=1)
	f = field(P, N, coeffs(model, P, N, nb))
	out = []
	for delta in (0.5, 1.0):  # outward probes (inward ones cross thin parts of the snail)
		ph, _ = f(P + delta * sig_p[:, None] * N)
		out.append((ph - delta * sig_p) / sig_p)
	rows['snail'] = {'out_abs': _q(np.abs(np.concatenate(out)))}
	gt = [rows[nm] for nm in rows if nm != 'snail']
	summ = dict(
		band_p999=max(x['band_abs'][3] for x in gt),
		band_max=max(x['band_abs'][4] for x in gt),
		band_p50=float(np.median([x['band_abs'][0] for x in gt])),
		far_p999=max(x['far_abs'][3] for x in gt if 'far_abs' in x),
		snail_out_p999=rows['snail']['out_abs'][3],
	)
	return summ, rows


# ---------------------------------------------------------------- 4. loss on fixed validation queries
def loss_eval(model, k, validation_file):
	import train as T

	P_all, N_all, nb_all, m_all, V_all, F_all, FM_all = T.load_mesh_training_data(validation_file)
	res = {}
	for knn in (0, 32):
		l1, eik = [], []
		for i in range(len(P_all)):
			P, N, nb, m = P_all[i].astype(np.float32), N_all[i].astype(np.float32), nb_all[i][:, :k], m_all[i]
			rows = np.random.default_rng(1000 + i).choice(np.nonzero(m)[0], min(256, int(m.sum())), replace=False)
			q, D = MS.generate_neighborhood_features(
				jnp.asarray(P[nb[rows]]), jnp.asarray(V_all[i], jnp.float32), jnp.asarray(F_all[i]), jnp.asarray(FM_all[i]),
				jax.random.PRNGKey(1000 + i),
			)
			C = coeffs(model, P, N, nb)
			mj = jnp.asarray(m)
			cen, ax, R, r = torus_precompute_masked(mj, jnp.asarray(P), jnp.asarray(N), jnp.asarray(C))
			for s in range(0, len(rows), 128):
				d, g = torus_distance_gradient_masked(mj, q[s : s + 128], jnp.asarray(P), cen, ax, R, r, knn=knn)
				l1.append(np.abs(np.asarray(d) - np.asarray(D[s : s + 128])).ravel())
				eik.append(np.abs(1 - np.linalg.norm(np.asarray(g), axis=-1)).ravel())
		tag = 'all' if knn == 0 else f'knn{knn}'
		res[f'val_l1_{tag}'] = float(np.nanmean(np.concatenate(l1)))
		res[f'val_eik_{tag}'] = float(np.nanmean(np.concatenate(eik)))
	return res


# ---------------------------------------------------------------- gate and report
GATES = [
	('kerr <= 0.1', lambda r: r['curv']['kerr'], lambda v: v <= 0.1),
	('analytic normal angle <= 4 deg', lambda r: r['surf']['surf_an_ang'], lambda v: v <= 4.0),
	('sphere kappa*rho within 30% of -1', lambda r: r['curv']['sph_kr'], lambda v: -1.3 <= v <= -0.7),
	('plane kappa*sigma < 0', lambda r: r['curv']['plane_ks'], lambda v: v < 0),
	('worst GT shape band p99.9 (sigma)', lambda r: r['err']['band_p999'], None),
	('worst GT shape band max (sigma)', lambda r: r['err']['band_max'], None),
	('snail outward p99.9 < 1 sigma', lambda r: r['err']['snail_out_p999'], lambda v: v < 1.0),
	('far p99.9 < 0.5 sigma (worst GT shape)', lambda r: r['err']['far_p999'], lambda v: v < 0.5),
]


def gate_report(res):
	"""Lines '[gate] check value PASS|MISS|(info)'; returns (lines, all passed)."""
	lines, ok_all = [], True
	for name, get, test in GATES:
		v = get(res)
		ok = None if test is None else bool(test(v))
		ok_all = ok_all and ok is not False
		lines.append(f'[gate] {name:42s} {v:9.4f}  {"(info)" if ok is None else ("PASS" if ok else "MISS")}')
	return lines, ok_all


def evaluate(path, label=None, out_dir=None, validation_file=None, loss=True):
	t0 = time.time()
	model, k = load_model(path)
	label = label or os.path.splitext(os.path.basename(path))[0]
	res = dict(model=os.path.abspath(path), label=label)
	res['curv'], res['curv_rows'] = curvature_tracking(model, k)
	res['surf'], res['surf_rows'] = surface_metrics(model, k)
	res['err'], res['err_rows'] = error_distributions(model, k)
	if loss and validation_file and os.path.exists(validation_file):
		res['loss'] = loss_eval(model, k, validation_file)
	res['seconds'] = time.time() - t0
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
		with open(os.path.join(out_dir, f'{label}.json'), 'w') as f:
			json.dump(res, f, indent=1)
	return res


HEAD = (
	f'{"label":28s} {"kerr":>6s} {"slope":>6s} {"aniso":>6s} {"plane":>6s} {"sphKr":>6s} {"inSad":>6s} | {"angAn":>5s} '
	f'{"angMe":>5s} {"lsH":>6s} | {"b999":>6s} {"bmax":>6s} {"far999":>6s} {"sn999":>6s} | {"valL1":>7s} {"valEik":>7s} {"s":>4s}'
)


def row(r):
	c, s, e, lo = r['curv'], r['surf'], r['err'], r.get('loss', {})
	f = lambda d, key, w=7, p=4: f'{d[key]:{w}.{p}f}' if key in d else f'{"-":>{w}s}'  # noqa: E731
	return (
		f'{r["label"][:28]:28s} {c["kerr"]:6.3f} {c["track_slope"]:6.3f} {c["aniso_slope"]:6.3f} {c["plane_ks"]:6.3f} '
		f'{c["sph_kr"]:6.2f} {c["inner_saddle"]:6.3f} | {s["surf_an_ang"]:5.2f} {s["surf_mesh_ang"]:5.2f} {s["surf_an_lsH_err"]:6.4f} | '
		f'{e["band_p999"]:6.3f} {e["band_max"]:6.3f} {e["far_p999"]:6.3f} {e["snail_out_p999"]:6.3f} | '
		f'{f(lo, "val_l1_all")} {f(lo, "val_eik_all")} {r["seconds"]:4.0f}'
	)


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('models', nargs='+', help='PATH or LABEL=PATH (.pkl from train.py, or .npz)')
	ap.add_argument('--out', default='eval', help='output directory for the JSON reports')
	ap.add_argument('--data-dir', default='datasets', help='training data directory (for the validation file)')
	ap.add_argument('--validation-file', default=None, help='default: DATA_DIR/ABC17_validation_3.npz')
	ap.add_argument('--no-loss', action='store_true', help='skip the loss on fixed validation queries')
	ap.add_argument('--gate', action='store_true', help='print the gate checks for each model')
	a = ap.parse_args(argv)
	vf = a.validation_file or os.path.join(a.data_dir, 'ABC17_validation_3.npz')
	print(HEAD, flush=True)
	for m in a.models:
		label, path = m.split('=', 1) if '=' in m else (None, m)
		r = evaluate(path, label, a.out, vf, loss=not a.no_loss)
		print(row(r), flush=True)
		if a.gate:
			for line in gate_report(r)[0]:
				print(line, flush=True)


if __name__ == '__main__':
	main()
