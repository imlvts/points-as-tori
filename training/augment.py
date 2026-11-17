"""
CPU-side batch construction for train.py's multi-shape patch batches (--multishape).

A patch of one shape: a seed point (uniform among the valid points), its n_centres nearest points as the query
centres, and the union U of their k-NN. The network runs on U only (each point of U with its own stored k-NN as
input), the tori of U are blended (masked to U), and the queries are generated per centre neighborhood. U is padded
to a fixed size (upad) so every step has the same shapes; if U is larger, the points of U nearest the seed are kept.

Per visit of a shape (visit), optionally: fresh area-uniform points on the ground-truth mesh at a random density
(--resample), with the k-NN recomputed; a random rotation of points, normals and mesh (--rotate); Gaussian noise on
the input points (--noise); directional quantization of the input points (--quantize); angular noise on the input
normals (--normal-noise). Perturbations change the input points only: normals and the ground-truth mesh stay exact,
so the network learns to fit the clean surface to perturbed input.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def random_rotation(rng):
	"""A uniformly random rotation matrix (3, 3)."""
	return Rotation.random(random_state=rng).as_matrix()
from scipy.spatial import cKDTree


def point_scale(P, tree, k):
	"""sigma of a cloud: the median over its points of the median distance to their k nearest neighbors."""
	d, _ = tree.query(P, k=k)
	return float(np.median(np.median(d, axis=1)))


def quantize_along(P, direction, step):
	"""Round the coordinate of every point along the unit vector direction to a multiple of step (the depth
	quantization of a scanner looking along direction); the two other coordinates are unchanged."""
	h = P @ direction
	return P + (step * np.round(h / step) - h)[:, None] * direction[None, :]


def perturb_normals(N, degrees, rng):
	"""Angular noise on unit normals: n + g projected to the tangent plane, g ~ N(0, (degrees in radians)^2 I),
	renormalized (each tangent component of the tilt has standard deviation ~degrees). Zero (padded) normals stay zero."""
	g = rng.normal(size=N.shape) * np.radians(degrees)
	g -= np.sum(g * N, axis=-1, keepdims=True) * N
	M = N + g
	norm = np.linalg.norm(M, axis=-1, keepdims=True)
	valid = np.linalg.norm(N, axis=-1, keepdims=True) > 0
	return np.where(valid, M / np.where(valid, norm, 1.0), N)


def visit(arrays, k, rng=None, resample='', rotate=False, noise=0.0, quantize=0.0, normal_noise=0.0, decimated_max=0):
	"""
	One visit of a stored shape -> (P, N, NB, M, V, F, FM, tree); tree is None when the stored k-NN are used.

	resample 'LO:HI': replace the stored points by n fresh area-uniform samples of the valid faces with their face
	normals, n drawn uniformly from the powers of two in [LO, HI]; the k-NN then come from tree (a cKDTree).
	rotate: apply one uniformly random rotation to the points, normals and mesh vertices (k-NN are unchanged).
	noise: add N(0, (noise sigma)^2) to every coordinate of the points, sigma the cloud's median k-NN distance (before
		the perturbation); padding is dropped and the k-NN are recomputed on the perturbed points.
	quantize: round the coordinate of the points along a random direction (one per visit) to multiples of
		quantize sigma (applied before the noise); padding is dropped and the k-NN are recomputed.
	normal_noise: angular noise of normal_noise degrees on the input normals (perturb_normals; k-NN unchanged).
	decimated_max: > 0: with resample, a mesh whose face count is the padded maximum (it was decimated to that
		limit, 2048 in the released data) is resampled to at most decimated_max points (the counts of resample up to
		it; the smallest one if none).
	"""
	P, N, NB, M, V, F, FM = arrays
	tree = None
	if resample:
		counts = resample_counts(resample)
		if decimated_max and FM.all():
			counts = [c for c in counts if c <= decimated_max] or counts[:1]
		n = int(rng.choice(counts))
		P, N = sample_mesh_points(V.astype(np.float64), F[FM], n, rng)
		M = np.ones(n, bool)
		NB = None
	else:
		NB = NB[:, :k]
	if rotate:
		R = random_rotation(rng)
		P, N, V = P @ R.T, N @ R.T, V @ R.T
	if noise or quantize:
		P, N = P[M], N[M]
		M = np.ones(len(P), bool)
		sigma = point_scale(P, cKDTree(P), k)
		if quantize:
			direction = rng.normal(size=3)
			direction /= np.linalg.norm(direction)
			P = quantize_along(P, direction, quantize * sigma)
		if noise:
			P = P + rng.normal(size=P.shape) * (noise * sigma)
		NB = None
	if normal_noise:
		N = perturb_normals(N, normal_noise, rng)
	if NB is None:
		tree = cKDTree(P)
	return P, N, NB, M, V, F, FM, tree


def make_patch(P, N, NB, M, tree, k, n_centres, upad, rng, point_data=None, hole=None, seed_area=False, union_k=0):
	"""
	Args:
		P, N: (n, 3) points and normals; NB: (n, k) neighbors or None (then tree, a cKDTree of P, gives them);
		M: (n,) valid-point mask; rng: numpy Generator.
		point_data: None, or (n, m) per-point values; then the result is (the tuple below, (upad, m) their values on
			U, padded like Upos)
		hole: None, or (p, r): with probability p, cut a hole next to the seed before the centres are chosen: the points
			within r sigma of P[seed] + (r + 0.5) sigma t are removed (t a random unit tangent at the seed, sigma the
			median distance of the seed's k-NN), so the hole's rim runs through the patch; the k-NN are recomputed.
			The ground truth (mesh / exact SDF) is not changed: the network learns to close the gap.
		seed_area: draw the seed with probability ~ s_i^2 (s_i: distance to the 8th nearest neighbour, s_i^2 ~ the
			surface area per point) instead of uniformly over the points, by sampling-importance-resampling from 64
			uniform candidates (exact in the limit, cheap on every cloud).
		union_k: > k: U also contains the union_k nearest points of every centre (a halo: the blend of a query near
			the patch rim then sees the points that are among its nearest in the whole cloud); the network inputs stay
			the k-NN. Points beyond upad are dropped farthest from the seed first, i.e. halo points first.

	Returns:
		UP, UN: (upad, k, 3) network inputs (neighborhoods of the points of U)
		Upos, Unrm: (upad, 3) the points of U (padded by repeating the first one)
		Umask: (upad,) valid entries of U
		centre_nbhds: (n_centres, k, 3) neighborhoods of the centres (for the query generator)
		u: the size of U before padding
	"""
	valid = np.nonzero(M)[0]
	if seed_area:
		cand = rng.choice(valid, 64)
		if NB is not None:
			s = np.linalg.norm(P[NB[cand, min(8, NB.shape[1] - 1)]] - P[cand], axis=-1)
		else:
			s = tree.query(P[cand], k=9)[0][:, -1]
		w = s**2
		seed = int(cand[rng.choice(64, p=w / w.sum())] if w.sum() > 0 else cand[0])
	else:
		seed = int(rng.choice(valid))
	if hole is not None and rng.uniform() < hole[0]:
		nbs = NB[seed] if NB is not None else tree.query(P[seed], k=k)[1]
		sigma = float(np.median(np.linalg.norm(P[nbs] - P[seed], axis=-1)))
		n = N[seed] / max(np.linalg.norm(N[seed]), 1e-12)
		t = rng.normal(size=3)
		t -= (t @ n) * n
		t /= max(np.linalg.norm(t), 1e-12)
		c = P[seed] + (hole[1] + 0.5) * sigma * t
		keep = np.asarray(M, bool) & (np.linalg.norm(P - c, axis=-1) > hole[1] * sigma)
		keep[seed] = True
		if keep.sum() >= 2 * k:  # (skipped on clouds too small to keep k-NN)
			idx = np.nonzero(keep)[0]
			P, N = P[idx], N[idx]
			if point_data is not None:
				point_data = point_data[idx]
			seed = int(np.searchsorted(idx, seed))
			M, NB, tree = np.ones(len(idx), bool), None, cKDTree(P)
	if NB is not None:
		centres = NB[seed][:n_centres]
		U = np.unique(NB[centres].ravel())
	else:
		centres = tree.query(P[seed], k=n_centres)[1]
		U = np.unique(tree.query(P[centres], k=k)[1].ravel())
	if union_k > k:
		if NB is not None:
			nv = np.nonzero(M)[0]
			halo = nv[cKDTree(P[nv]).query(P[centres], k=min(union_k, len(nv)))[1].ravel()]
		else:
			halo = tree.query(P[centres], k=min(union_k, len(P)))[1].ravel()
		U = np.unique(np.concatenate([U, halo]))
	U = U[U >= 0]
	U = U[M[U]]
	u = len(U)
	if u > upad:
		U = U[np.argsort(np.linalg.norm(P[U] - P[seed], axis=1))[:upad]]
	nbU = NB[U] if NB is not None else tree.query(P[U], k=k)[1]
	nbC = NB[centres] if NB is not None else tree.query(P[centres], k=k)[1]
	pad = upad - len(U)
	Uf = np.concatenate([U, np.repeat(U[:1], pad)])
	nbUf = np.concatenate([nbU, np.repeat(nbU[:1], pad, 0)])
	mask = np.arange(upad) < len(U)
	f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
	out = f32(P[nbUf]), f32(N[nbUf]), f32(P[Uf]), f32(N[Uf]), mask, f32(P[nbC]), u
	if point_data is not None:
		return out, f32(point_data[Uf])
	return out


def resample_counts(spec):
	"""'LO:HI' -> the powers of two in [LO, HI] (one compiled program per count)."""
	lo, hi = (int(x) for x in spec.split(':'))
	counts = [2**e for e in range(4, 20) if lo <= 2**e <= hi]
	if not counts:
		raise ValueError(f'--resample {spec}: no power of two in range')
	return counts


def sample_mesh_points(V, F, n, rng):
	"""n area-uniform points on the triangles F of V, with the face normals."""
	a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
	f = rng.choice(len(F), n, p=a / a.sum())
	u, v = rng.uniform(size=n), rng.uniform(size=n)
	flip = u + v > 1
	u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
	P = V[F[f, 0]] + u[:, None] * (V[F[f, 1]] - V[F[f, 0]]) + v[:, None] * (V[F[f, 2]] - V[F[f, 0]])
	Nf = np.cross(V[F[f, 1]] - V[F[f, 0]], V[F[f, 2]] - V[F[f, 0]])
	return P, Nf / np.linalg.norm(Nf, axis=1, keepdims=True)
