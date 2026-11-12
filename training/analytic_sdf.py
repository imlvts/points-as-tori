"""Exact SDFs of analytic scenes (analytic_scene.py, gen_analytic.py) and the query generator of train.py --mix.

A shape = up to MP primitives, pairwise disjoint, so the SDF of the union is the min of the primitive SDFs (exact
everywhere: outside, the nearest primitive; inside a primitive, its own distance). Encoding (float32 row of 16):
  type 0 sphere   [cx cy cz R]
  type 1 capsule  [ax ay az bx by bz r]
  type 2 torus    [cx cy cz ux uy uz R rho]                (axis u unit)
  type 3 rbox     [cx cy cz qw qx qy qz bx by bz rr]       (inner half-extents b, rounding rr; rotation quaternion)
  type 4 tube     [poly_id r]                              (union of capsules along a dense polyline of a smooth curve;
                                                            exact SDF of that capsule chain; sagitta < 1e-3 sigma)
  type 5 revseg   [cx cy cz ux uy uz r0 y0 r1 y1 t]        (thin wall of revolution: meridian segment thickened by t;
                                                            pipe / cone shell / washer)
  type 6 bowl     [cx cy cz ux uy uz r h t]                (sphere radius r cut at height h, part below, thickened by t)
  type -1 none
Polylines: (MT, NS, 3) with a valid-point count per polyline.
Rotation augmentation: the cloud is P @ R.T; queries q are evaluated as sdf(q @ R).
"""
from functools import partial

import jax
import jax.numpy as jnp

MP, MT, NS = 8, 3, 512


def _quat_rot(q):
	w, x, y, z = q[0], q[1], q[2], q[3]
	return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
					  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
					  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _seg_dist(q, a, b):
	ba = b - a
	t = jnp.clip(jnp.sum((q - a) * ba, -1) / jnp.maximum(jnp.sum(ba * ba, -1), 1e-20), 0.0, 1.0)
	return jnp.linalg.norm(q - (a + t[..., None] * ba), axis=-1)


def _prim_sdf(q, p, typ, poly_d):
	"""q (m,3) -> (m,) SDF of one primitive; poly_d (m, MT) distances to the polylines."""
	sph = jnp.linalg.norm(q - p[:3], axis=-1) - p[3]
	cap = _seg_dist(q, p[0:3], p[3:6]) - p[6]
	u = p[3:6] / jnp.maximum(jnp.linalg.norm(p[3:6]), 1e-12)
	r_ = q - p[:3]
	h = jnp.sum(r_ * u, -1)
	rad = jnp.linalg.norm(r_ - h[:, None] * u, axis=-1)
	tor = jnp.sqrt((rad - p[6]) ** 2 + h ** 2) - p[7]
	Rm = _quat_rot(p[3:7] / jnp.maximum(jnp.linalg.norm(p[3:7]), 1e-12))
	loc = (q - p[:3]) @ Rm  # world -> box frame (columns of Rm are the box axes)
	qq = jnp.abs(loc) - p[7:10]
	# safe norm: inside a box max(qq, 0) = 0 and the plain norm's gradient is NaN (through the jnp.where below it
	# would poison the gradient of every query inside any box)
	box = jnp.sqrt(jnp.sum(jnp.maximum(qq, 0.0) ** 2, -1) + 1e-30) + jnp.minimum(jnp.max(qq, -1), 0.0) - p[10]
	pid = jnp.clip(p[0].astype(jnp.int32), 0, MT - 1)
	tub = poly_d[:, pid] - p[1]
	# revolved thin walls (types 5, 6): meridian coords (rad, h) about axis u through p[:3] (computed above)
	A2, B2 = p[6:8], p[8:10]
	Q2 = jnp.stack([rad, h], -1)
	ba = B2 - A2
	s = jnp.clip(jnp.sum((Q2 - A2) * ba, -1) / jnp.maximum(jnp.sum(ba * ba), 1e-20), 0.0, 1.0)
	rseg = jnp.linalg.norm(Q2 - (A2 + s[:, None] * ba), axis=-1) - p[10]
	br, bh = p[6], p[7]
	bw = jnp.sqrt(jnp.maximum(br * br - bh * bh, 0.0))
	rim = jnp.sqrt((rad - bw) ** 2 + (h - bh) ** 2)
	bowl = jnp.where(bh * rad < bw * h, rim, jnp.abs(jnp.sqrt(rad ** 2 + h ** 2) - br)) - p[8]
	out = jnp.where(typ == 0, sph, jnp.where(typ == 1, cap, jnp.where(typ == 2, tor, jnp.where(typ == 3, box,
					jnp.where(typ == 4, tub, jnp.where(typ == 5, rseg, bowl))))))
	return jnp.where(typ < 0, jnp.inf, out)


# The one-shot min over all 511 segments compiles to one fully unrolled reduction kernel (~96k PTX lines; ptxas
# takes tens of minutes). A scan over chunks of POLY_CHUNK segments with a running min of the squared distance gives
# the same values with small kernels. 0 = the one-shot path.
POLY_CHUNK = 32


def _poly_dist_chunked(q, polys, seg_ok):
	"""(m, 3) x (MT, NS, 3) -> (m, MT) distance to each polyline (valid segments only)."""
	A, B = polys[:, :-1], polys[:, 1:]  # (MT, NS-1, 3)
	nseg = A.shape[1]
	pad = (-nseg) % POLY_CHUNK
	A = jnp.pad(A, ((0, 0), (0, pad), (0, 0)), mode='edge')
	B = jnp.pad(B, ((0, 0), (0, pad), (0, 0)), mode='edge')
	ok = jnp.pad(seg_ok, ((0, 0), (0, pad)), constant_values=False)
	nc = A.shape[1] // POLY_CHUNK
	A = A.reshape(A.shape[0], nc, POLY_CHUNK, 3).transpose(1, 0, 2, 3)  # (nc, MT, C, 3)
	B = B.reshape(B.shape[0], nc, POLY_CHUNK, 3).transpose(1, 0, 2, 3)
	ok = ok.reshape(ok.shape[0], nc, POLY_CHUNK).transpose(1, 0, 2)

	def body(best, xs):
		a, b, o = xs
		ba = b - a  # (MT, C, 3)
		qa = q[:, None, None, :] - a[None]  # (m, MT, C, 3)
		t = jnp.clip(jnp.sum(qa * ba[None], -1) / jnp.maximum(jnp.sum(ba * ba, -1), 1e-20)[None], 0.0, 1.0)
		d2 = jnp.sum((qa - t[..., None] * ba[None]) ** 2, -1)  # (m, MT, C)
		return jnp.minimum(best, jnp.min(jnp.where(o[None], d2, jnp.inf), -1)), None
	best, _ = jax.lax.scan(body, jnp.full((q.shape[0], polys.shape[0]), jnp.inf, q.dtype), (A, B, ok))
	return jnp.sqrt(best)


def sdf(q, prims, types, polys, pcount):
	"""Exact SDF of the scene at q (m, 3)."""
	seg_ok = jnp.arange(NS - 1)[None, :] < (pcount[:, None] - 1)  # (MT, NS-1)

	if POLY_CHUNK:
		poly_d = _poly_dist_chunked(q, polys, seg_ok)
	else:
		def poly_dist(P, ok):
			d = _seg_dist(q[:, None, :], P[None, :-1], P[None, 1:])  # (m, NS-1)
			return jnp.min(jnp.where(ok[None], d, jnp.inf), axis=-1)
		poly_d = jax.vmap(poly_dist, out_axes=1)(polys, seg_ok)  # (m, MT)
	d = jax.vmap(lambda p, t: _prim_sdf(q, p, t, poly_d))(prims, types)  # (MP, m)
	return jnp.min(d, axis=0)


def sdf_grad(q, prims, types, polys, pcount):
	f = lambda x: sdf(x[None], prims, types, polys, pcount)[0]  # noqa: E731
	return jax.vmap(jax.value_and_grad(f))(q)


def _boxes(nbp, bbox_extension):
	bmin, bmax = nbp.min(1), nbp.max(1)
	c = 0.5 * (bmin + bmax)
	ext = jnp.maximum(jnp.abs(bmax - bmin), 0.1)
	side = jnp.maximum(jnp.max(jnp.abs(bmax - bmin), axis=-1), 0.01)
	rad = 0.5 * jnp.sqrt(3.0) * side * bbox_extension
	return c - 0.5 * ext, c + 0.5 * ext, c - rad[:, None], c + rad[:, None]


@partial(jax.jit, static_argnames=['n_queries', 'n_cand'])
def analytic_queries(nbp, prims, types, polys, pcount, R, key, bbox_extension=1.0, n_queries=120, n_cand=512):
	"""The local query sampler (mesh_sampling_jax.sample_zero_set_and_band_local) with an exact-SDF ground truth (no mesh). Per neighbourhood (rows of nbp, world frame):
    zero-set = uniform candidates in the regularised box projected onto the surface (x - phi grad phi, exact closest
    point for an exact SDF), kept if inside the box; band = zero + U(-0.2, 0.2) * grad phi, kept if inside the cube box;
    cube = uniform in the cube box. SDF exact. R: rotation used for the cloud (world = shape @ R.T)."""
	m = n_queries // 3
	rlo, rhi, clo, chi = _boxes(nbp, bbox_extension)
	keys = jax.random.split(key, nbp.shape[0])

	def phi_g(x):  # world-frame values and gradients
		v, g = sdf_grad(x @ R, prims, types, polys, pcount)
		return v, g @ R.T

	def pick(ok, kk, n):
		r = jax.random.uniform(kk, ok.shape)
		order = jnp.argsort(jnp.where(ok, r, 2.0 + r))
		na = jnp.sum(ok)
		return order[jnp.arange(n) % jnp.maximum(na, 1)], na

	def one(lo, hi, cl, ch, pts, kk):
		k1, k2, k3, k4, k5 = jax.random.split(kk, 5)
		x = lo + jax.random.uniform(k1, (n_cand, 3)) * (hi - lo)
		v, g = phi_g(x)
		z = x - v[:, None] * g
		z = z - phi_g(z)[0][:, None] * g  # second projection step (exact already, guards rounding)
		okz = jnp.all((z >= lo) & (z <= hi), -1) & jnp.all(jnp.isfinite(z), -1)
		iz, nz = pick(okz, k2, m)
		_, gz = phi_g(z)
		y = z + jax.random.uniform(k3, (n_cand,), minval=-0.2, maxval=0.2)[:, None] * gz
		okb = okz & jnp.all((y >= cl) & (y <= ch), -1)
		ib, nb_ = pick(okb, k4, m)
		fall = pts[jnp.arange(m) % pts.shape[0]]
		zq = jnp.where(nz > 0, z[iz], fall)
		bq = jnp.where(nb_ > 0, y[ib], fall)
		cq = cl + jax.random.uniform(k5, (n_queries - 2 * m, 3)) * (ch - cl)
		return jnp.concatenate([zq, bq, cq], 0), nz == 0

	q, f0 = jax.vmap(one)(rlo, rhi, clo, chi, nbp, keys)
	d = jax.vmap(lambda qq: sdf(qq @ R, prims, types, polys, pcount))(q)
	return q, d, jnp.mean(f0)
