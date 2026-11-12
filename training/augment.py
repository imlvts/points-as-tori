"""
CPU-side batch construction for train.py's multi-shape patch batches (--multishape).

A patch of one shape: a seed point (uniform among the valid points), its n_centres nearest points as the query
centres, and the union U of their k-NN. The network runs on U only (each point of U with its own stored k-NN as
input), the tori of U are blended (masked to U), and the queries are generated per centre neighborhood. U is padded
to a fixed size (upad) so every step has the same shapes; if U is larger, the points of U nearest the seed are kept.
"""

import numpy as np
from scipy.spatial import cKDTree


def visit(arrays, k):
	"""One visit of a stored shape -> (P, N, NB, M, V, F, FM, tree); tree is None when the stored k-NN are used."""
	P, N, NB, M, V, F, FM = arrays
	return P, N, NB[:, :k], M, V, F, FM, None


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
