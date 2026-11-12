"""
CPU-side batch construction for train.py's multi-shape patch batches (--multishape).

A patch of one shape: a seed point (uniform among the valid points), its n_centres nearest points as the query
centres, and the union U of their k-NN. The network runs on U only (each point of U with its own stored k-NN as
input), the tori of U are blended (masked to U), and the queries are generated per centre neighborhood. U is padded
to a fixed size (upad) so every step has the same shapes; if U is larger, the points of U nearest the seed are kept.

Per visit of a shape (visit), optionally: fresh area-uniform points on the ground-truth mesh at a random density
(--resample), with the k-NN recomputed; a random rotation of points, normals and mesh (--rotate).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def random_rotation(rng):
	"""A uniformly random rotation matrix (3, 3)."""
	return Rotation.random(random_state=rng).as_matrix()
from scipy.spatial import cKDTree


def visit(arrays, k, rng=None, resample='', rotate=False):
	"""
	One visit of a stored shape -> (P, N, NB, M, V, F, FM, tree); tree is None when the stored k-NN are used.

	resample 'LO:HI': replace the stored points by n fresh area-uniform samples of the valid faces with their face
	normals, n drawn uniformly from the powers of two in [LO, HI]; the k-NN then come from tree (a cKDTree).
	rotate: apply one uniformly random rotation to the points, normals and mesh vertices (k-NN are unchanged).
	"""
	P, N, NB, M, V, F, FM = arrays
	tree = None
	if resample:
		n = int(rng.choice(resample_counts(resample)))
		P, N = sample_mesh_points(V.astype(np.float64), F[FM], n, rng)
		M = np.ones(n, bool)
		NB = None
	else:
		NB = NB[:, :k]
	if rotate:
		R = random_rotation(rng)
		P, N, V = P @ R.T, N @ R.T, V @ R.T
	if NB is None:
		tree = cKDTree(P)
	return P, N, NB, M, V, F, FM, tree


def make_patch(P, N, NB, M, tree, k, n_centres, upad, rng):
	"""
	Args:
		P, N: (n, 3) points and normals; NB: (n, k) neighbors or None (then tree, a cKDTree of P, gives them);
		M: (n,) valid-point mask; rng: numpy Generator.

	Returns:
		UP, UN: (upad, k, 3) network inputs (neighborhoods of the points of U)
		Upos, Unrm: (upad, 3) the points of U (padded by repeating the first one)
		Umask: (upad,) valid entries of U
		centre_nbhds: (n_centres, k, 3) neighborhoods of the centres (for the query generator)
		u: the size of U before padding
	"""
	valid = np.nonzero(M)[0]
	seed = int(rng.choice(valid))
	if NB is not None:
		centres = NB[seed][:n_centres]
		U = np.unique(NB[centres].ravel())
	else:
		centres = tree.query(P[seed], k=n_centres)[1]
		U = np.unique(tree.query(P[centres], k=k)[1].ravel())
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
	return f32(P[nbUf]), f32(N[nbUf]), f32(P[Uf]), f32(N[Uf]), mask, f32(P[nbC]), u


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
