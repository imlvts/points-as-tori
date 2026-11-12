"""Analytic scenes (numpy): primitives, area-uniform surface sampling with exact normals and principal curvatures, a
numpy SDF (placement checks, verification), and the scene families used by train.py --mix. Encoding: analytic_sdf.py.

Curvature sign: convex with outward normal < 0 (as eval_model.curvatures). k1 = the larger |kappa|.
Point classes (ptype): 0 sphere / capsule cap / tube end cap, 1 tube or capsule side (closed thin tube when r/sigma is
small), 2 torus outer half, 3 torus inner half (saddle: k_tube < 0 < k_ring), 4 box face (flat), 5 box edge (fillet),
6 box corner.
"""
import numpy as np
from scipy.spatial import cKDTree

MP, MT, NS = 8, 3, 512


# ------------------------------------------------------------------ helpers
def unit(v):
	return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rand_rot(rng):
	q = rng.normal(size=4)
	q /= np.linalg.norm(q)
	return q


def quat_rot(q):
	w, x, y, z = q
	return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
					 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
					 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def seg_dist(q, a, b):
	ba = b - a
	t = np.clip(((q - a) @ ba) / max(ba @ ba, 1e-20), 0, 1)
	return np.linalg.norm(q - (a + t[:, None] * ba), axis=-1)


def poly_dist(q, P, chunk=4096):
	out = np.empty(len(q))
	A, B = P[:-1], P[1:]
	BA = B - A
	L2 = np.maximum(np.sum(BA * BA, -1), 1e-20)
	for s in range(0, len(q), chunk):
		x = q[s:s + chunk, None, :]
		t = np.clip(np.sum((x - A) * BA, -1) / L2, 0, 1)
		out[s:s + chunk] = np.min(np.linalg.norm(x - (A + t[..., None] * BA), axis=-1), axis=1)
	return out


# ------------------------------------------------------------------ primitives (dicts)
def prim_sdf(p, q):
	t = p['type']
	if t == 'sphere':
		return np.linalg.norm(q - p['c'], axis=-1) - p['R']
	if t == 'capsule':
		return seg_dist(q, p['a'], p['b']) - p['r']
	if t == 'torus':
		r_ = q - p['c']
		h = r_ @ p['u']
		rad = np.linalg.norm(r_ - h[:, None] * p['u'], axis=-1)
		return np.sqrt((rad - p['R']) ** 2 + h ** 2) - p['rho']
	if t == 'rbox':
		loc = (q - p['c']) @ quat_rot(p['q'])
		qq = np.abs(loc) - p['b']
		return np.linalg.norm(np.maximum(qq, 0), axis=-1) + np.minimum(qq.max(-1), 0) - p['rr']
	if t == 'tube':
		return poly_dist(q, p['P']) - p['r']
	if t in ('revseg', 'bowl'):
		rho, y = merid(p, q)
		if t == 'revseg':
			A, B = np.array([p['r0'], p['y0']]), np.array([p['r1'], p['y1']])
			Q2 = np.stack([rho, y], 1)
			ba = B - A
			s = np.clip(((Q2 - A) @ ba) / max(ba @ ba, 1e-20), 0, 1)
			return np.linalg.norm(Q2 - (A + s[:, None] * ba), axis=-1) - p['t']
		r, h = p['r'], p['h']
		w = np.sqrt(r * r - h * h)
		rim = np.sqrt((rho - w) ** 2 + (y - h) ** 2)
		return np.where(h * rho < w * y, rim, np.abs(np.sqrt(rho ** 2 + y ** 2) - r)) - p['t']
	raise ValueError(t)


def merid(p, q):
	"""Meridian coordinates (distance from the axis, height along u) of a surface of revolution about (c, u)."""
	r_ = q - p['c']
	y = r_ @ p['u']
	return np.linalg.norm(r_ - y[:, None] * p['u'], axis=-1), y


def merid_pieces(p):
	"""Meridian boundary of a revolved thin wall: list of (length, f(s in [0,1]) -> (rho, y, n_rho, n_y, k_meridian)).
    revseg = segment (r0, y0)-(r1, y1) thickened by t (two straight sides + two semicircular rims);
    bowl = sphere radius r cut at height h (the part below), thickened by t (outer / inner zones + one rim)."""
	t = p['t']
	if p['type'] == 'revseg':
		A, B = np.array([p['r0'], p['y0']]), np.array([p['r1'], p['y1']])
		L = np.linalg.norm(B - A)
		d = (B - A) / L
		n = np.array([-d[1], d[0]])
		out = []
		for sg in (1, -1):
			def side(s, sg=sg):
				x = A + s[:, None] * (B - A) + sg * t * n
				nn = np.broadcast_to(sg * n, x.shape)
				return x[:, 0], x[:, 1], nn[:, 0], nn[:, 1], np.zeros(len(s))
			out.append((L, side))
		for C, dd in ((A, -d), (B, d)):
			def cap(s, C=C, dd=dd):
				b = np.pi * s - 0.5 * np.pi  # -pi/2..pi/2 around the outward direction dd
				nn = np.cos(b)[:, None] * dd + np.sin(b)[:, None] * n
				x = C + t * nn
				return x[:, 0], x[:, 1], nn[:, 0], nn[:, 1], np.full(len(s), -1 / t)
			out.append((np.pi * t, cap))
		return out
	r, h = p['r'], p['h']
	a0 = np.arccos(h / r)
	er, ea = np.array([np.sin(a0), np.cos(a0)]), np.array([np.cos(a0), -np.sin(a0)])
	out = []
	for sg in (1, -1):
		def zone(s, sg=sg):
			a = a0 + s * (np.pi - a0)
			nn = np.stack([np.sin(a), np.cos(a)], 1)
			x = (r + sg * t) * nn
			return x[:, 0], x[:, 1], sg * nn[:, 0], sg * nn[:, 1], np.full(len(s), -sg / (r + sg * t))
		out.append(((r + sg * t) * (np.pi - a0), zone))

	def rim(s):
		b = np.pi * s  # e_r (outer side) .. -e_r (inner side) through -e_alpha (beyond the rim)
		nn = np.cos(b)[:, None] * er - np.sin(b)[:, None] * ea
		x = r * er + t * nn
		return x[:, 0], x[:, 1], nn[:, 0], nn[:, 1], np.full(len(s), -1 / t)
	out.append((np.pi * t, rim))
	return out


def rev_frame(p):
	a, b = frame_of(p['u'])
	return a, b


def sample_rev(p, n, rng):
	"""Area-uniform samples of a revolved thin wall (rejection on the meridian with weight rho), exact normals and
    principal curvatures (meridian k_m; parallel k_p = -n_rho / rho). ptype 7 wall side, 8 wall rim."""
	pieces = merid_pieces(p)
	L = np.array([x[0] for x in pieces])
	xmax = 0.0
	for _, f in pieces:
		xmax = max(xmax, float(f(np.linspace(0, 1, 257))[0].max()))
	rows = []
	got = 0
	while got < n:
		m = 4 * (n - got) + 64
		which = rng.choice(len(pieces), m, p=L / L.sum())
		for w in range(len(pieces)):
			sel = which == w
			if not sel.any():
				continue
			s = rng.uniform(0, 1, int(sel.sum()))
			rho, y, nr, ny, km = pieces[w][1](s)
			keep = rng.uniform(size=len(s)) * xmax < rho
			rows.append((rho[keep], y[keep], nr[keep], ny[keep], km[keep], np.full(int(keep.sum()), w)))
			got += int(keep.sum())
	cols = [np.concatenate([r_[i] for r_ in rows]) for i in range(6)]
	perm = rng.permutation(len(cols[0]))[:n]  # rows are grouped by piece: shuffle before truncating
	rho, y, nr, ny, km, wid = (c_[perm] for c_ in cols)
	phi = rng.uniform(0, 2 * np.pi, n)
	a, b = rev_frame(p)
	e = np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b
	P = p['c'] + rho[:, None] * e + y[:, None] * p['u']
	N = nr[:, None] * e + ny[:, None] * p['u']
	kp = -nr / np.maximum(rho, 1e-12)
	rimw = {2, 3} if p['type'] == 'revseg' else {2}
	pt = np.where(np.isin(wid, list(rimw)), 8, 7).astype(np.int8)
	return P, N, km, kp, pt


def scene_sdf(prims, q):
	return np.min(np.stack([prim_sdf(p, q) for p in prims]), axis=0)


def frame_of(u):
	a = unit(np.cross(u, [0.31, 0.57, 0.76]))
	return a, np.cross(u, a)


def sample_prim(p, n, rng):
	"""n area-uniform points -> P, N, k1, k2, ptype (exact)."""
	t = p['type']
	if t in ('revseg', 'bowl'):
		return sample_rev(p, n, rng)
	if t == 'sphere':
		x = unit(rng.normal(size=(n, 3)))
		k = np.full(n, -1 / p['R'])
		return p['c'] + p['R'] * x, x, k, k.copy(), np.zeros(n, np.int8)
	if t in ('capsule', 'tube'):
		if t == 'capsule':
			C = np.stack([p['a'], p['b']])
		else:
			C = p['P']
		r = p['r']
		seg = np.diff(C, axis=0)
		L = np.linalg.norm(seg, axis=-1)
		Ltot = L.sum()
		a_side, a_cap = 2 * np.pi * r * Ltot, 4 * np.pi * r * r
		n_side = rng.binomial(n, a_side / (a_side + a_cap))
		# side: arc length uniform, theta with density (1 - r k cos(theta)) (rejection); Frenet frame from the polyline
		s_cum = np.concatenate([[0], np.cumsum(L)])
		T = unit(np.gradient(C, s_cum, axis=0)) if len(C) > 2 else np.repeat(unit(seg), 2, 0)
		if len(C) > 2:
			dT = np.gradient(T, s_cum, axis=0)
			kc = np.linalg.norm(dT, axis=-1)
			Nc = np.where(kc[:, None] > 1e-9, dT / np.maximum(kc[:, None], 1e-12), frame_of(T[0])[0])
		else:
			kc = np.zeros(2)
			Nc = np.repeat(frame_of(T[0])[0][None], 2, 0)
		Nc = unit(Nc - np.sum(Nc * T, -1, keepdims=True) * T)
		Bc = np.cross(T, Nc)
		out_s, out_th = [], []
		while sum(len(x) for x in out_s) < n_side:
			s = rng.uniform(0, Ltot, 4 * n_side + 16)
			th = rng.uniform(0, 2 * np.pi, len(s))
			k = np.interp(s, s_cum, kc)
			keep = rng.uniform(size=len(s)) < (1 - r * k * np.cos(th)) / (1 + r * kc.max())
			out_s.append(s[keep])
			out_th.append(th[keep])
		s, th = np.concatenate(out_s)[:n_side], np.concatenate(out_th)[:n_side]
		Cp = np.stack([np.interp(s, s_cum, C[:, i]) for i in range(3)], 1)
		Tp = unit(np.stack([np.interp(s, s_cum, T[:, i]) for i in range(3)], 1))
		Np_ = np.stack([np.interp(s, s_cum, Nc[:, i]) for i in range(3)], 1)
		Np_ = unit(Np_ - np.sum(Np_ * Tp, -1, keepdims=True) * Tp)
		Bp = np.cross(Tp, Np_)
		k = np.interp(s, s_cum, kc)
		nrm = np.cos(th)[:, None] * Np_ + np.sin(th)[:, None] * Bp
		P1 = Cp + r * nrm
		ka = np.cos(th) * k / (1 - r * k * np.cos(th))  # along the curve (> 0 on the inner side)
		kr = np.full(n_side, -1 / r)
		# end caps: hemispheres
		n_cap = n - n_side
		x = unit(rng.normal(size=(n_cap, 3)))
		end = rng.integers(0, 2, n_cap)
		tang = np.where(end[:, None] == 0, -T[0], T[-1])
		x = np.where(np.sum(x * tang, -1, keepdims=True) < 0, -x, x)
		ctr = np.where(end[:, None] == 0, C[0], C[-1])
		P2 = ctr + r * x
		kk = np.full(n_cap, -1 / r)
		k1 = np.concatenate([kr, kk])
		k2 = np.concatenate([ka, kk])
		return (np.concatenate([P1, P2]), np.concatenate([nrm, x]), k1, k2,
				np.concatenate([np.ones(n_side, np.int8), np.zeros(n_cap, np.int8)]))
	if t == 'torus':
		R, rho = p['R'], p['rho']
		a, b = frame_of(p['u'])
		us, vs = [], []
		while sum(len(x) for x in us) < n:
			u = rng.uniform(0, 2 * np.pi, 4 * n)
			v = rng.uniform(0, 2 * np.pi, 4 * n)
			keep = rng.uniform(size=4 * n) < (R + rho * np.cos(v)) / (R + rho)
			us.append(u[keep])
			vs.append(v[keep])
		u, v = np.concatenate(us)[:n], np.concatenate(vs)[:n]
		e = np.cos(u)[:, None] * a + np.sin(u)[:, None] * b
		nrm = np.cos(v)[:, None] * e + np.sin(v)[:, None] * p['u']
		P = p['c'] + R * e + rho * nrm
		kt = np.full(n, -1 / rho)
		kr = -np.cos(v) / (R + rho * np.cos(v))
		return P, nrm, kt, kr, np.where(np.cos(v) < 0, 3, 2).astype(np.int8)
	if t == 'rbox':
		Rm = quat_rot(p['q'])
		b, rr = p['b'], p['rr']
		faces = [(i, s) for i in range(3) for s in (-1, 1)]
		af = [4 * b[(i + 1) % 3] * b[(i + 2) % 3] for i, _ in faces]
		edges = [(i, sj, sk) for i in range(3) for sj in (-1, 1) for sk in (-1, 1)]  # edge along axis i
		ae = [0.5 * np.pi * rr * 2 * b[i] for i, _, _ in edges]
		ac = [0.5 * np.pi * rr * rr] * 8  # 8 eighth-spheres = 4 pi rr^2
		areas = np.array(af + ae + ac)
		which = rng.choice(len(areas), n, p=areas / areas.sum())
		P, Nn, k1, k2, pt = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n, np.int8)
		for w in np.unique(which):
			m = which == w
			cnt = int(m.sum())
			if w < 6:
				i, s = faces[w]
				loc = rng.uniform(-1, 1, (cnt, 3)) * b
				nn = np.zeros((cnt, 3))
				nn[:, i] = s
				loc[:, i] = s * b[i]
				pos = loc + rr * nn
				P[m], Nn[m], pt[m] = pos, nn, 4
			elif w < 18:
				i, sj, sk = edges[w - 6]
				j, k = (i + 1) % 3, (i + 2) % 3
				ang = rng.uniform(0, 0.5 * np.pi, cnt)
				nn = np.zeros((cnt, 3))
				nn[:, j] = sj * np.cos(ang)
				nn[:, k] = sk * np.sin(ang)
				base = np.zeros((cnt, 3))
				base[:, i] = rng.uniform(-b[i], b[i], cnt)
				base[:, j], base[:, k] = sj * b[j], sk * b[k]
				P[m], Nn[m], k1[m], pt[m] = base + rr * nn, nn, -1 / rr, 5
			else:
				sg = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])[w - 18]
				x = np.abs(unit(rng.normal(size=(cnt, 3)))) * sg
				P[m], Nn[m], k1[m], k2[m], pt[m] = sg * b + rr * x, x, -1 / rr, -1 / rr, 6
		if b.min() <= 1e-9:  # thin plate (one zero half-extent, rr = half-thickness): wall side 7, rim 8
			pt = np.where(pt == 4, 7, 8).astype(np.int8)
		return p['c'] + P @ Rm.T, Nn @ Rm.T, k1, k2, pt
	raise ValueError(t)


def prim_area(p):
	t = p['type']
	if t == 'sphere':
		return 4 * np.pi * p['R'] ** 2
	if t == 'capsule':
		return 2 * np.pi * p['r'] * np.linalg.norm(p['b'] - p['a']) + 4 * np.pi * p['r'] ** 2
	if t == 'tube':
		return 2 * np.pi * p['r'] * np.linalg.norm(np.diff(p['P'], axis=0), axis=-1).sum() + 4 * np.pi * p['r'] ** 2
	if t == 'torus':
		return 4 * np.pi ** 2 * p['R'] * p['rho']
	if t == 'rbox':
		b, rr = p['b'], p['rr']
		return 8 * (b[0] * b[1] + b[1] * b[2] + b[0] * b[2]) + 2 * np.pi * rr * (b.sum() * 2) + 4 * np.pi * rr * rr
	if t in ('revseg', 'bowl'):  # Pappus, trapezoid on a dense meridian grid
		s = np.linspace(0, 1, 4097)
		return float(sum(2 * np.pi * Lp * np.trapezoid(f(s)[0], s) for Lp, f in merid_pieces(p)))
	raise ValueError(t)


def sample_scene(prims, n, rng):
	a = np.array([prim_area(p) for p in prims])
	cnt = rng.multinomial(n, a / a.sum())
	parts = [sample_prim(p, c, rng) for p, c in zip(prims, cnt) if c > 0]
	pid = np.concatenate([np.full(c, i, np.int8) for i, c in enumerate(cnt) if c > 0])
	out = [np.concatenate([x[i] for x in parts]) for i in range(5)]
	swap = np.abs(out[3]) > np.abs(out[2])  # k1 = larger |kappa|
	k1 = np.where(swap, out[3], out[2])
	k2 = np.where(swap, out[2], out[3])
	return out[0].astype(np.float32), unit(out[1]).astype(np.float32), k1.astype(np.float32), k2.astype(np.float32), out[4], pid


def sigma_of(P, k=64):
	nb = cKDTree(P).query(P, k=k)[1]
	return float(np.median(np.median(np.linalg.norm(P[nb] - P[:, None], axis=-1), axis=1)))


# ------------------------------------------------------------------ encoding (analytic_sdf.py layout)
def encode(prims):
	E = np.zeros((MP, 16), np.float32)
	T = np.full(MP, -1, np.int32)
	polys = np.zeros((MT, NS, 3), np.float32)
	pc = np.zeros(MT, np.int32)
	nt = 0
	for i, p in enumerate(prims):
		t = p['type']
		if t == 'sphere':
			E[i, :4], T[i] = [*p['c'], p['R']], 0
		elif t == 'capsule':
			E[i, :7], T[i] = [*p['a'], *p['b'], p['r']], 1
		elif t == 'torus':
			E[i, :8], T[i] = [*p['c'], *p['u'], p['R'], p['rho']], 2
		elif t == 'rbox':
			E[i, :11], T[i] = [*p['c'], *p['q'], *p['b'], p['rr']], 3
		elif t == 'tube':
			polys[nt, :len(p['P'])] = p['P']
			polys[nt, len(p['P']):] = p['P'][-1]
			pc[nt] = len(p['P'])
			E[i, :2], T[i] = [nt, p['r']], 4
			nt += 1
		elif t == 'revseg':
			E[i, :11], T[i] = [*p['c'], *p['u'], p['r0'], p['y0'], p['r1'], p['y1'], p['t']], 5
		elif t == 'bowl':
			E[i, :9], T[i] = [*p['c'], *p['u'], p['r'], p['h'], p['t']], 6
	return E, T, polys, pc


# ------------------------------------------------------------------ families
def curve(kind, rng, length, r):
	"""Dense smooth curve (NS points) of about the given length, curvature radius >= 3.3 r."""
	s = np.linspace(0, 1, NS)
	if kind == 'arc':
		r0 = max(3.3 * r, length / (1.5 * np.pi))
		Rc = rng.uniform(r0, max(1.2 * r0, length / 0.8))
		ang = length / Rc
		P = np.stack([Rc * np.cos(ang * s), Rc * np.sin(ang * s), np.zeros(NS)], 1)
	elif kind == 'helix':
		turns = rng.uniform(0.6, 2.0)
		a0 = max(3.3 * r, 0.08)
		a = rng.uniform(a0, max(0.35, 1.5 * a0))
		pitch = max(rng.uniform(2.5, 5) * r + 0.02, 0.08)
		h = turns * pitch
		P = np.stack([a * np.cos(2 * np.pi * turns * s), a * np.sin(2 * np.pi * turns * s), h * s], 1)
		kc = a / (a * a + (pitch / (2 * np.pi)) ** 2)
		if r * kc > 0.3:
			return curve('arc', rng, length, r)
	else:  # 'bent': smooth random Bezier-like curve, then curvature check
		ctrl = np.cumsum(rng.normal(size=(4, 3)) * length / 3, 0)
		tt = s[:, None]
		P = ((1 - tt) ** 3 * ctrl[0] + 3 * (1 - tt) ** 2 * tt * ctrl[1] + 3 * (1 - tt) * tt ** 2 * ctrl[2] + tt ** 3 * ctrl[3])
		L = np.linalg.norm(np.diff(P, axis=0), axis=-1)
		sc = np.concatenate([[0], np.cumsum(L)])
		T = unit(np.gradient(P, sc, axis=0))
		k = np.linalg.norm(np.gradient(T, sc, axis=0), axis=-1)
		if r * k.max() > 0.3:
			return curve('arc', rng, length, r)
	# resample to uniform arc length
	L = np.linalg.norm(np.diff(P, axis=0), axis=-1)
	sc = np.concatenate([[0], np.cumsum(L)])
	su = np.linspace(0, sc[-1], NS)
	return np.stack([np.interp(su, sc, P[:, i]) for i in range(3)], 1)


def place(prim, prims, rng, gap, half=0.9, tries=60):
	"""Random rotation + translation so prim is inside [-half, half]^3 and >= gap away from prims; None if impossible."""
	base = prim
	for _ in range(tries):
		R = quat_rot(rand_rot(rng))
		p = dict(base)
		X, _, _, _, _ = sample_prim(base, 400, rng)
		Xr = X @ R.T
		lo, hi = Xr.min(0), Xr.max(0)
		if np.any(hi - lo > 2 * half):
			return None
		t = rng.uniform(-half - lo, half - hi)
		p = transform(base, R, t)
		Xs = Xr + t
		if all(np.min(prim_sdf(q, Xs)) > gap for q in prims):
			# also: no existing primitive's samples inside / near the new one
			ok = True
			for q in prims:
				Y, _, _, _, _ = sample_prim(q, 300, rng)
				if np.min(prim_sdf(p, Y)) <= gap:
					ok = False
					break
			if ok:
				return p
	return None


def transform(p, R, t):
	q = dict(p)
	if p['type'] == 'sphere':
		q['c'] = R @ p['c'] + t
	elif p['type'] == 'capsule':
		q['a'], q['b'] = R @ p['a'] + t, R @ p['b'] + t
	elif p['type'] in ('torus', 'revseg', 'bowl'):
		q['c'], q['u'] = R @ p['c'] + t, R @ p['u']
	elif p['type'] == 'rbox':
		Rm = R @ quat_rot(p['q'])
		q['c'] = R @ p['c'] + t
		q['q'] = mat_quat(Rm)
	elif p['type'] == 'tube':
		q['P'] = p['P'] @ R.T + t
	return q


def mat_quat(M):
	w = np.sqrt(max(0.0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
	x = np.sqrt(max(0.0, 1 + M[0, 0] - M[1, 1] - M[2, 2])) / 2
	y = np.sqrt(max(0.0, 1 - M[0, 0] + M[1, 1] - M[2, 2])) / 2
	z = np.sqrt(max(0.0, 1 - M[0, 0] - M[1, 1] + M[2, 2])) / 2
	x = np.copysign(x, M[2, 1] - M[1, 2])
	y = np.copysign(y, M[0, 2] - M[2, 0])
	z = np.copysign(z, M[1, 0] - M[0, 1])
	q = np.array([w, x, y, z])
	return q / np.linalg.norm(q)


def body(rng, size):
	kind = rng.choice(['sphere', 'rbox', 'torus'], p=[0.4, 0.4, 0.2])
	if kind == 'sphere':
		return dict(type='sphere', c=np.zeros(3), R=size)
	if kind == 'rbox':
		b = rng.uniform(0.25, 1.0, 3) * size
		return dict(type='rbox', c=np.zeros(3), q=np.array([1.0, 0, 0, 0]), b=b, rr=rng.uniform(0.03, 0.4) * size)
	R = rng.uniform(0.55, 0.8) * size
	return dict(type='torus', c=np.zeros(3), u=np.array([0, 0, 1.0]), R=R, rho=rng.uniform(0.2, 0.6) * R)


def make_scene(family, rng, n=2048, tube_r=None, wall_t=(0.15, 1.0)):
	"""-> prims (list of dicts), meta. Thin radii are set in units of a sigma estimated from the body's area.
    tube_r: (lo, hi) tube radius range in sigma (log-uniform; None = the default (0.2, 1.5), and uniform in 'mixed');
    wall_t: half-thickness range of thin walls in sigma (log-uniform; families 'walls' and 'sticks')."""
	prims = []
	tr = tube_r or (0.2, 1.5)
	meta = {'family': family}
	s_est = lambda A: 3.1 * np.sqrt(A / n)  # noqa: E731  (calibrated on spheres: sigma = 0.197 at A = 8.04, n = 2048)
	if family in ('tubes', 'spheres', 'mixed'):
		size = rng.uniform(0.35, 0.6)
		p = None
		while p is None:
			p = place(body(rng, size), [], rng, 0.0)
			size *= 0.85
		prims.append(p)
	if family == 'tubes':
		A = sum(prim_area(p) for p in prims)
		for _ in range(rng.integers(1, 5)):
			sig = s_est(A)
			r = np.exp(rng.uniform(np.log(tr[0]), np.log(tr[1]))) * sig  # log-uniform: more weight on thin tubes
			kind = rng.choice(['straight', 'arc', 'helix', 'bent', 'ring'])
			if kind == 'ring':  # thin closed ring = exact bent tube (torus), radius ratio >= 4
				Rr = max(4 * r, 0.1) * rng.uniform(1.0, 2.5)
				p = place(dict(type='torus', c=np.zeros(3), u=np.array([0, 0, 1.0]), R=Rr, rho=r), prims, rng, gap=max(2 * sig, 0.05))
				if p is not None and len(prims) < MP:
					prims.append(p)
				A = sum(prim_area(x) for x in prims)
				continue
			length = rng.uniform(0.4, 1.4)
			if kind == 'straight':
				tp = dict(type='capsule', a=np.zeros(3), b=np.array([0, 0, length]), r=r)
			else:
				tp = dict(type='tube', P=curve(kind, rng, length, r), r=r)
			p = place(tp, prims, rng, gap=max(2 * sig, 0.05))
			if p is not None and (p['type'] != 'tube' or sum(x['type'] == 'tube' for x in prims) < MT) and len(prims) < MP:
				prims.append(p)
			A = sum(prim_area(x) for x in prims)
	elif family == 'tori':
		for _ in range(rng.integers(1, 3)):
			ratio = np.exp(rng.uniform(np.log(1.3), np.log(10)))
			R = rng.uniform(0.25, 0.7)
			tp = dict(type='torus', c=np.zeros(3), u=np.array([0, 0, 1.0]), R=R, rho=R / ratio)
			p = place(tp, prims, rng, gap=0.08)
			if p is not None:
				prims.append(p)
		if not prims:
			prims.append(dict(type='torus', c=np.zeros(3), u=np.array([0, 0, 1.0]), R=0.6, rho=0.2))
	elif family == 'spheres':
		A = prim_area(prims[0])
		for _ in range(rng.integers(2, 7)):
			sig = s_est(A)
			rad = sig * np.exp(rng.uniform(np.log(0.3), np.log(8)))
			tp = dict(type='sphere', c=np.zeros(3), R=rad) if rng.uniform() < 0.6 else \
				dict(type='capsule', a=np.zeros(3), b=np.array([0, 0, rng.uniform(0.1, 0.6)]), r=rad)
			p = place(tp, prims, rng, gap=max(2 * sig, 0.05))
			if p is not None:
				prims.append(p)
			A = sum(prim_area(x) for x in prims)
	elif family == 'boxes':
		for _ in range(rng.integers(1, 3)):
			size = rng.uniform(0.3, 0.75)
			b = rng.uniform(0.15, 1.0, 3) * size
			sig = s_est(8 * (b[0] * b[1] + b[1] * b[2] + b[0] * b[2]) + 1e-3)
			rr = sig * np.exp(rng.uniform(np.log(0.3), np.log(5)))
			p = place(dict(type='rbox', c=np.zeros(3), q=np.array([1.0, 0, 0, 0]), b=b, rr=rr), prims, rng, gap=0.08)
			if p is not None:
				prims.append(p)
		if not prims:
			prims.append(dict(type='rbox', c=np.zeros(3), q=np.array([1.0, 0, 0, 0]), b=np.array([0.5, 0.3, 0.2]), rr=0.05))
	elif family == 'mixed':
		A = prim_area(prims[0])
		for _ in range(rng.integers(2, 6)):
			sig = s_est(A)
			kind = rng.choice(['sphere', 'capsule', 'tube', 'torus', 'rbox'])
			if kind == 'sphere':
				tp = dict(type='sphere', c=np.zeros(3), R=sig * np.exp(rng.uniform(np.log(0.3), np.log(6))))
			elif kind == 'capsule':
				rc = rng.uniform(0.2, 1.5) if tube_r is None else np.exp(rng.uniform(np.log(tr[0]), np.log(tr[1])))
				tp = dict(type='capsule', a=np.zeros(3), b=np.array([0, 0, rng.uniform(0.2, 0.9)]), r=rc * sig)
			elif kind == 'tube':
				r = rng.uniform(0.2, 1.5) * sig if tube_r is None else np.exp(rng.uniform(np.log(tr[0]), np.log(tr[1]))) * sig
				tp = dict(type='tube', P=curve(rng.choice(['arc', 'helix', 'bent']), rng, rng.uniform(0.4, 1.2), r), r=r)
			elif kind == 'torus':
				R = rng.uniform(0.15, 0.4)
				tp = dict(type='torus', c=np.zeros(3), u=np.array([0, 0, 1.0]), R=R, rho=R / np.exp(rng.uniform(np.log(1.5), np.log(10))))
			else:
				tp = dict(type='rbox', c=np.zeros(3), q=np.array([1.0, 0, 0, 0]), b=rng.uniform(0.05, 0.3, 3),
						  rr=rng.uniform(0.3, 3) * sig)
			if kind == 'tube' and sum(x['type'] == 'tube' for x in prims) >= MT:
				continue
			p = place(tp, prims, rng, gap=max(2 * sig, 0.05))
			if p is not None and len(prims) < MP:
				prims.append(p)
			A = sum(prim_area(x) for x in prims)
	elif family == 'walls':
		# thin walls: flat plates (rbox with one zero half-extent), pipes, cone shells, washers (revseg), bowls;
		# optionally next to a body. Half-thickness t log-uniform in wall_t sigma, both faces 2t apart.
		if rng.uniform() < 0.4:
			size = rng.uniform(0.3, 0.5)
			p = None
			while p is None:
				p = place(body(rng, size), [], rng, 0.0)
				size *= 0.85
			prims.append(p)
		U = lambda lo, hi: rng.uniform(lo, max(hi, 1.3 * lo))  # noqa: E731  (thick walls: keep the minimum ratios)
		A = sum(prim_area(p) for p in prims) + 0.6  # + about one wall, so sigma is not overestimated for the first one
		for _ in range(rng.integers(1, 4)):
			sig = s_est(A)
			t = sig * np.exp(rng.uniform(np.log(wall_t[0]), np.log(wall_t[1])))
			kind = rng.choice(['plate', 'pipe', 'cone', 'washer', 'bowl'], p=[0.3, 0.2, 0.15, 0.1, 0.25])
			z3, ez = np.zeros(3), np.array([0, 0, 1.0])
			if kind == 'plate':
				tp = dict(type='rbox', c=z3, q=np.array([1.0, 0, 0, 0]), b=np.array([*rng.uniform(0.1, 0.6, 2), 0.0]), rr=t)
			elif kind == 'pipe':
				Rc, H = U(max(4 * t, 0.06), 0.4), rng.uniform(0.1, 0.5)
				tp = dict(type='revseg', c=z3, u=ez, r0=Rc, y0=-H, r1=Rc, y1=H, t=t)
			elif kind == 'cone':
				ra, rb, H = U(max(3 * t, 0.04), 0.4), U(max(3 * t, 0.04), 0.4), rng.uniform(0.1, 0.4)
				tp = dict(type='revseg', c=z3, u=ez, r0=ra, y0=-H, r1=rb, y1=H, t=t)
			elif kind == 'washer':
				ra = U(max(3 * t, 0.03), 0.25)
				tp = dict(type='revseg', c=z3, u=ez, r0=ra, y0=0.0, r1=ra + rng.uniform(0.08, 0.4), y1=0.0, t=t)
			else:
				r = U(max(5 * t, 0.12), 0.5)
				hr = rng.uniform(-0.7, 0.85)
				if np.sqrt(1 - hr * hr) * r < 3 * t:
					hr = 0.5
				tp = dict(type='bowl', c=z3, u=ez, r=r, h=hr * r, t=t)
			p = place(tp, prims, rng, gap=max(2 * sig, 0.05))
			if p is not None and len(prims) < MP:
				prims.append(p)
			A = sum(prim_area(x) for x in prims)
		if not prims:
			prims.append(dict(type='rbox', c=np.zeros(3), q=np.array([1.0, 0, 0, 0]), b=np.array([0.4, 0.3, 0.0]), rr=0.02))
	elif family == 'sticks':
		# only thin sticks, no body: 2-7 capsules (75 %) and bent tubes. sigma comes from the sticks' own area, so the
		# radius solves r = x s_est(2 pi r L) (side area dominates): r = x^2 9.61 * 2 pi L / n.
		# The 64-NN sigma of a near-line cloud is larger than this estimate: a scene whose measured r / sigma (2048
		# points) is below 0.08 is redrawn (up to 8 times).
		for _attempt in range(8):
			prims = []
			k = int(rng.integers(2, 8))
			Ls = rng.uniform(0.3, 1.4, k)
			x = np.exp(rng.uniform(np.log(tr[0]), np.log(tr[1])))  # radius / sigma, one per scene
			r = x * x * 9.61 * 2 * np.pi * Ls.sum() / n
			sig = r / x
			for L in Ls:
				if rng.uniform() < 0.75 or sum(q['type'] == 'tube' for q in prims) >= MT:
					tp = dict(type='capsule', a=np.zeros(3), b=np.array([0, 0, L]), r=r)
				else:
					tp = dict(type='tube', P=curve(rng.choice(['arc', 'bent']), rng, L, r), r=r)
				p = place(tp, prims, rng, gap=max(2 * sig, 0.03))
				if p is not None and len(prims) < MP:
					prims.append(p)
			if prims and r / sigma_of(sample_scene(prims, n, rng)[0]) >= 0.08:
				break
		if not prims:
			prims.append(dict(type='capsule', a=np.array([0, 0, -0.5]), b=np.array([0, 0, 0.5]), r=0.01))
		meta['stick_r_over_sigma'] = float(x)
	return prims[:MP], meta


def decode(E, T, polys, pc):
	"""Encoded row (analytic_sdf layout) -> list of primitive dicts."""
	prims = []
	for p, t in zip(E, T):
		p = p.astype(np.float64)
		if t == 0:
			prims.append(dict(type='sphere', c=p[:3], R=float(p[3])))
		elif t == 1:
			prims.append(dict(type='capsule', a=p[:3], b=p[3:6], r=float(p[6])))
		elif t == 2:
			prims.append(dict(type='torus', c=p[:3], u=p[3:6], R=float(p[6]), rho=float(p[7])))
		elif t == 3:
			prims.append(dict(type='rbox', c=p[:3], q=p[3:7], b=p[7:10], rr=float(p[10])))
		elif t == 4:
			pid = int(p[0])
			prims.append(dict(type='tube', P=polys[pid][:pc[pid]].astype(np.float64), r=float(p[1])))
		elif t == 5:
			prims.append(dict(type='revseg', c=p[:3], u=p[3:6] / np.linalg.norm(p[3:6]), r0=float(p[6]), y0=float(p[7]),
							  r1=float(p[8]), y1=float(p[9]), t=float(p[10])))
		elif t == 6:
			prims.append(dict(type='bowl', c=p[:3], u=p[3:6] / np.linalg.norm(p[3:6]), r=float(p[6]), h=float(p[7]), t=float(p[8])))
	return prims
