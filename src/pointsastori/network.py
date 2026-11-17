import os
import gc
import subprocess
import glob
import numpy as np
import scipy as sp
import time
import pickle
import random

import jax
import jax.numpy as jnp
from jax import jit
from flax import nnx
from functools import partial
import optax

from typing import Tuple, List

from .shape_3d import *

jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_debug_nans', False)

NAN_EPSILON = 1e-10

MODELS_DIR = 'models/'

# ========================================================================
# HELPERS
# ========================================================================


def sign_floor(x: jnp.ndarray, eps: float = NAN_EPSILON) -> jnp.ndarray:
	"""x with |x| floored to eps, keeping its sign; +eps at x = 0 (the C++ sign(), shape_2d.h: sign(0) = +1).
	jnp.sign(0) = 0 would turn the guard sign(x) * max(|x|, eps) into 0 at x = 0."""
	return jnp.where(x >= 0, 1.0, -1.0) * jnp.maximum(jnp.abs(x), eps)


@jit
def orthonormal_basis_with_rotation(n: jnp.ndarray, rotation: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Compute an orthonormal basis (s, t) perpendicular to normal vector n, with a specified rotation of the axes.

	Args:
		n: (..., 3) array of normal vectors (will be normalized)
		rotation: scalar rotation angle about the normal

	Returns:
		s_hat: (..., 3) first basis vector
		t_hat: (..., 3) second basis vector
	"""
	# Normalize the normal vector
	n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True)
	n_hat = n / jnp.maximum(n_norm, NAN_EPSILON)

	# Choose initial vector u based on which component of n is smallest
	# If |n.x| > 0.9, use (0,1,0), otherwise use (1,0,0)
	abs_n_x = jnp.abs(n_hat[..., 0:1])
	u = jnp.where(
		abs_n_x > 0.9,
		jnp.concatenate([jnp.zeros_like(abs_n_x), jnp.ones_like(abs_n_x), jnp.zeros_like(abs_n_x)], axis=-1),
		jnp.concatenate([jnp.ones_like(abs_n_x), jnp.zeros_like(abs_n_x), jnp.zeros_like(abs_n_x)], axis=-1),
	)

	# Apply rotation about n_hat using Rodrigues' formula
	# R = I + sin(θ) * K + (1 - cos(θ)) * K^2
	# where K is the skew-symmetric matrix for n_hat
	cos_rot = jnp.cos(rotation)
	sin_rot = jnp.sin(rotation)

	# Compute u_rotated = cos(θ)*u + sin(θ)*(n × u) + (1-cos(θ))*(n·u)*n
	n_cross_u = jnp.cross(n_hat, u)
	n_dot_u = jnp.sum(n_hat * u, axis=-1, keepdims=True)
	u_rotated = cos_rot * u + sin_rot * n_cross_u + (1.0 - cos_rot) * n_dot_u * n_hat

	# Compute s = n × u (perpendicular to both)
	s = jnp.cross(n_hat, u_rotated)
	s_norm = jnp.linalg.norm(s, axis=-1, keepdims=True)
	s_hat = s / jnp.maximum(s_norm, NAN_EPSILON)

	# Compute t = n × s (completes orthonormal basis)
	t_hat = jnp.cross(n_hat, s_hat)

	return s_hat, t_hat


# ------------------------------------------------------------------------
# Tangent frames of the network input (FundamentalFormPredictor frame='continuous').
#
# The six coefficients are always read in the frame above (the "legacy" frame, also the C++ fit_torus_at_point's), whose
# reference vector switches at |n_x| = 0.9: on that circle the in-plane axes jump. A model with another input frame
# computes its features there and rotates its prediction into the legacy frame (rotate_coeffs), so the torus fits of
# training and inference are unchanged and the predicted torus no longer depends on the legacy frame.
#
# No tangent frame field on the sphere is continuous everywhere (hairy ball theorem). The frame below (Frisvad 2012,
# the frame of the rotation that takes the pole axis to n along a great circle) is continuous and smooth except at a
# single direction, the pole. Duff et al. 2017's branchless variant is NOT used: its copysign(1, n_z) makes the axes
# jump by 180 degrees across the whole equator n_z = 0. The pole is put at a direction that is not a likely CAD
# face normal (not axis-aligned, not a face diagonal).
# ------------------------------------------------------------------------

_POLE = np.array([-1.0, -2.0, -3.0]) / np.sqrt(14.0)
_E3 = -_POLE
_E1 = np.cross(_E3, [1.0, 0.0, 0.0])
_E1 /= np.linalg.norm(_E1)
_E2 = np.cross(_E3, _E1)
# Rows: a right-handed basis whose third axis is -pole; POLE_ROTATION @ pole = (0, 0, -1).
POLE_ROTATION = np.stack([_E1, _E2, _E3]).astype(np.float32)


def continuous_frame(n: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Orthonormal (s, t) perpendicular to the unit vectors n (..., 3), right-handed with t = n x s like the legacy frame,
	continuous (smooth) in n except at n = pole. In coordinates m = POLE_ROTATION n (pole at m = -z) this is Frisvad's
	frame s = (1 - m_x^2 c, -m_x m_y c, -m_x), c = 1 / (1 + m_z), with 1 + m_z = (m_x^2 + m_y^2) / (1 - m_z) for m_z < 0
	(no cancellation near the pole).
	"""
	Q = jnp.asarray(POLE_ROTATION, n.dtype)
	n_hat = n / jnp.maximum(jnp.linalg.norm(n, axis=-1, keepdims=True), NAN_EPSILON)
	m = jnp.einsum('ij,...j->...i', Q, n_hat)
	mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
	rho2 = mx * mx + my * my
	one_plus_mz = jnp.where(mz >= 0, 1.0 + mz, rho2 / (1.0 - mz))
	at_pole = one_plus_mz < 1e-30
	c = 1.0 / jnp.where(at_pole, 1.0, one_plus_mz)
	s_m = jnp.stack([1.0 - mx * mx * c, -mx * my * c, -mx], axis=-1)
	s_m = jnp.where(at_pole[..., None], jnp.array([1.0, 0.0, 0.0], n.dtype), s_m)
	s = jnp.einsum('ji,...j->...i', Q, s_m)  # back to world coordinates: Q^T s_m
	s = s - jnp.sum(s * n_hat, axis=-1, keepdims=True) * n_hat
	s = s / jnp.maximum(jnp.linalg.norm(s, axis=-1, keepdims=True), NAN_EPSILON)
	t = jnp.cross(n_hat, s)
	return s, t


def invariant_frames(positions: jnp.ndarray, s_hat: jnp.ndarray, t_hat: jnp.ndarray):
	"""
	Frames of the weighted frame averaging of FundamentalFormPredictor frame='invariant' (Dym, Lawrence, Siegel 2024,
	"Equivariant frame averaging"; continuous canonicalization is impossible, weighted averaging over a small
	equivariant frame set is not). With z_j = s_j + i t_j the tangent coordinates of the neighbours relative to the
	centre in a base frame (s_hat, t_hat) ([batch, 1, 3]), the spin-1 and spin-2 moments m1 = sum z_j, m2 = sum z_j^2
	rotate with the frame, so the three frames at angles arg m1, arg(m2)/2, arg(m2)/2 + pi (relative to the base frame)
	are attached to the neighbourhood, not to the base frame. Weights (sum 1): |m1|^2 / (sum |z_j|)^2 for the first,
	|m2|^2 / (sum |z_j|^2)^2 / 2 for each of the others, i.e. a frame counts in proportion to how well its moment
	determines it, and an ill-determined angle (moment -> 0) has weight -> 0: the average is continuous in the input
	and exactly invariant to the base frame. (Only if both moments vanish exactly the weights fall back to equal
	weights of frames that are then not determined by the neighbourhood.)

	Returns: angles [batch, 3] (relative to the base frame), weights [batch, 3].
	"""
	d = positions - positions[:, 0:1, :]
	x = jnp.sum(d * s_hat, axis=-1)  # [batch, k]
	y = jnp.sum(d * t_hat, axis=-1)
	r2 = x * x + y * y
	m1x, m1y = jnp.sum(x, axis=-1), jnp.sum(y, axis=-1)
	m2x, m2y = jnp.sum(x * x - y * y, axis=-1), jnp.sum(2 * x * y, axis=-1)
	tiny = 1e-30
	w1 = (m1x * m1x + m1y * m1y) / jnp.maximum(jnp.sum(jnp.sqrt(r2), axis=-1) ** 2, tiny)
	w2 = (m2x * m2x + m2y * m2y) / jnp.maximum(jnp.sum(r2, axis=-1) ** 2, tiny)
	th1 = jnp.arctan2(m1y, m1x)
	th2 = 0.5 * jnp.arctan2(m2y, m2x)
	angles = jnp.stack([th1, th2, th2 + jnp.pi], axis=-1)
	weights = jnp.stack([w1, 0.5 * w2, 0.5 * w2], axis=-1) + 1e-12
	return angles, weights / jnp.sum(weights, axis=-1, keepdims=True)


def aux_blend_terms(aux: jnp.ndarray, hard: bool = False):
	"""
	--aux-heads logits [..., 2] -> (point_sign, log_weight) of torus_distance_gradient_masked.
	point_sign: the torus' SDF is negated (the point's normal is taken as flipped) where the flip logit is < 0. In
	training a straight-through estimate: the value is exactly +-1 (as at inference), the gradient that of tanh.
	log_weight = log sigmoid(confidence logit) <= 0: the log of the point's confidence alpha in (0, 1), which multiplies
	its blend weight (only ratios matter: the blend is normalized). hard: no straight-through term (inference).
	"""
	z_flip, z_conf = aux[..., 0], aux[..., 1]
	sign = jnp.where(z_flip >= 0, 1.0, -1.0)
	if not hard:
		soft = jnp.tanh(z_flip)
		sign = sign + soft - jax.lax.stop_gradient(soft)
	return sign, jax.nn.log_sigmoid(z_conf)


def rotate_coeffs(c: jnp.ndarray, cos_phi: jnp.ndarray, sin_phi: jnp.ndarray) -> jnp.ndarray:
	"""
	Coefficients (a00, a01, a10, a11, a02, a20) of the height h = a00 + a10 s' + a01 t' + a20 s'^2 + a11 s' t' + a02 t'^2
	in the frame s' = cos_phi s + sin_phi t, t' = -sin_phi s + cos_phi t -> the same quadric in the frame (s, t).
	(The gradient (a10, a01) rotates as a vector, the Hessian as a symmetric tensor; a00 is invariant.)
	"""
	a00, a01, a10, a11, a02, a20 = [c[..., i] for i in range(6)]
	co, si = cos_phi, sin_phi
	b10 = co * a10 - si * a01
	b01 = si * a10 + co * a01
	b20 = a20 * co**2 - a11 * co * si + a02 * si**2
	b11 = 2 * co * si * (a20 - a02) + (co**2 - si**2) * a11
	b02 = a20 * si**2 + a11 * co * si + a02 * co**2
	return jnp.stack([a00, b01, b10, b11, b02, b20], axis=-1)


# ========================================================================
# (AUTO-DIFFERENTIABLE) DISTANCE FUNCTIONS
# ========================================================================


@jit
def cartesian_to_rotation_jax(u: jnp.ndarray) -> jnp.ndarray:
	"""
	Convert 3D unit vector to spherical coordinates (theta, phi).

	Args:
		u: (..., 3) array of 3D vectors

	Returns:
		axes: (..., 2) array of (theta, phi) in radians
			theta: azimuthal angle in [0, 2π]
			phi: polar angle in [0, π]
	"""
	# Normalize input
	u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
	u_hat = u / jnp.maximum(u_norm, NAN_EPSILON)

	x = u_hat[..., 0]
	y = u_hat[..., 1]
	z = u_hat[..., 2]

	phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
	safe_x = sign_floor(x)
	theta = jnp.arctan2(y, safe_x)
	return jnp.stack([theta, phi], axis=-1)


@jit
def rotation_to_cartesian_jax(axes: jnp.ndarray) -> jnp.ndarray:
	"""
	Convert spherical coordinates (theta, phi) to 3D unit vector.

	Args:
		axes: (..., 2) array of (theta, phi) in radians

	Returns:
		u: (..., 3) array of 3D unit vectors
	"""
	theta = axes[..., 0]
	phi = axes[..., 1]

	# Spherical to Cartesian conversion
	sin_phi = jnp.sin(phi)
	cos_phi = jnp.cos(phi)
	sin_theta = jnp.sin(theta)
	cos_theta = jnp.cos(theta)

	x = sin_phi * cos_theta
	y = sin_phi * sin_theta
	z = cos_phi

	return jnp.stack([x, y, z], axis=-1)


def _compute_shifts_from_distances(distances: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
	"""
	Compute per-query shift from a pre-computed distance matrix.

	Args:
		distances: (batch_size, n_queries, max_points) distances from each query to each point
		mask: (max_points,) valid-point mask

	Returns:
		shifts: (batch_size, n_queries)
	"""
	k = 64
	max_points = distances.shape[-1]
	k_actual = min(k, max_points)

	m = mask[None, None, :]  # (1, 1, max_points)

	# top_k on negated distances: invalid points → -inf so they're never selected.
	neg_distances = jnp.where(m, -distances, -jnp.inf)
	neg_knn, _ = jax.lax.top_k(neg_distances, k_actual)  # (batch, n_queries, k_actual)
	knn_dists = -neg_knn  # positive, ascending order (nearest first)

	# Shift = 0.5 * max distance among k nearest valid neighbors
	finite_knn = jnp.where(jnp.isfinite(knn_dists), knn_dists, 0.0)
	return 0.5 * jnp.max(finite_knn, axis=-1)  # (batch, n_queries)


@jit
def compute_lambdas_jax(shifts: jnp.ndarray) -> jnp.ndarray:
	"""
	Compute lambda parameter from shift.

	Args:
		shift: (...,) shift values

	Returns:
		lam: (...,) lambda values
	"""
	return 64.0 / shifts


def _masked_fit_inputs(mask, points, normals, coeffs):
	"""The inputs of the torus fits with the masked rows replaced by a dummy."""
	# Masked (padded) rows get a fixed, well-conditioned dummy input (unit normal, the network's initial sphere
	# coefficients) before the fit. The where passes no cotangent to the masked rows, so a degenerate fit there (zero
	# normals, arbitrary coefficients) can neither produce NaN values nor NaN gradients; the blend drops these tori
	# through its mask as before.
	row_mask = mask[:, None]
	points = jnp.where(row_mask, points, 0.0)
	normals = jnp.where(row_mask, normals, jnp.array([0.0, 0.0, 1.0], dtype=normals.dtype))
	coeffs = jnp.where(row_mask, coeffs, jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5], dtype=coeffs.dtype))
	return points, normals, coeffs


def _principal_fit(p_i, n_i, c_i):
	"""The quadric fit at one point: (a00, kappa_min, kappa_max, v_min, normal) with |kappa_min| <= |kappa_max|, v_min
	the (unnormalized) principal direction of kappa_min and normal = n*(0,0), the unit normal at the contact point
	p* = p_i + a00 n_i (for unit n_i)."""
	# Compute orthonormal bases
	s_hat, t_hat = orthonormal_basis_with_rotation(n_i, 0.0)

	# Extract polynomial coefficients
	a00 = c_i[0]
	a01 = c_i[1]
	a10 = c_i[2]
	a11 = c_i[3]
	a02 = c_i[4]
	a20 = c_i[5]

	# Compute principal curvatures from polynomial fit
	A = jnp.sqrt(1.0 + a01**2 + a10**2)
	# A = jnp.maximum(A, NAN_EPSILON)

	# Compute discriminant for eigenvector calculation
	v1 = a20 * a01**2 - a02 * a10**2 - a02 + a20
	v2 = a11 * (a10**2 + 1.0) - 2.0 * a01 * a10 * a20
	term = a02 * (a10**2 + 1.0) - a01 * a10 * a11 + a20 * (a01**2 + 1.0)
	discriminant = (1.0 + a01**2 + a10**2) * (a11**2 - 4.0 * a02 * a20) + term**2
	discriminant = jnp.sqrt(jnp.maximum(discriminant, NAN_EPSILON))  # sqrt not differentiable at 0

	ev_pos_0 = v1 + discriminant
	ev_pos_0 = sign_floor(ev_pos_0)
	ev_pos = jnp.array([ev_pos_0, v2])

	ev_neg_1 = v2
	ev_neg_1 = sign_floor(ev_neg_1)
	ev_neg = jnp.array([v1 - discriminant, ev_neg_1])

	# Even though normalization isn't necessary because we take a Rayleigh quotient, it helps numerically.
	ev_pos /= jnp.linalg.norm(ev_pos)
	ev_neg /= jnp.linalg.norm(ev_neg)
	ev_pos_x = ev_pos[0]
	ev_pos_y = ev_pos[1]
	ev_neg_x = ev_neg[0]
	ev_neg_y = ev_neg[1]

	# Convert to 3D eigenvectors
	dQstards = s_hat + a10 * n_i  # (3, )
	dQstardt = t_hat + a01 * n_i  # (3, )

	v_pos = ev_pos_x * dQstards + ev_pos_y * dQstardt
	v_neg = ev_neg_x * dQstards + ev_neg_y * dQstardt

	# Compute normal at fitted point
	normal = (n_i - a10 * s_hat - a01 * t_hat) / A

	# Compute curvatures using Rayleigh quotient
	I_11 = 1.0 + a10**2
	I_12 = a10 * a01
	I_22 = 1.0 + a01**2

	II_11 = 2.0 * a20 / A
	II_12 = a11 / A
	II_22 = 2.0 * a02 / A

	# Compute kappa = (ev^T @ II @ ev) / (ev^T @ I @ ev)
	# For positive eigenvector
	II_ev_pos_x = II_11 * ev_pos_x + II_12 * ev_pos_y
	II_ev_pos_y = II_12 * ev_pos_x + II_22 * ev_pos_y
	I_ev_pos_x = I_11 * ev_pos_x + I_12 * ev_pos_y
	I_ev_pos_y = I_12 * ev_pos_x + I_22 * ev_pos_y

	numer_pos = ev_pos_x * II_ev_pos_x + ev_pos_y * II_ev_pos_y
	denom_pos = ev_pos_x * I_ev_pos_x + ev_pos_y * I_ev_pos_y
	denom_pos = sign_floor(denom_pos)
	kappa_pos = numer_pos / denom_pos
	# kappa_pos = jax.lax.cond(numer_pos == 0, lambda: NAN_EPSILON, lambda: numer_pos / denom_pos)

	# For negative eigenvector
	II_ev_neg_x = II_11 * ev_neg_x + II_12 * ev_neg_y
	II_ev_neg_y = II_12 * ev_neg_x + II_22 * ev_neg_y
	I_ev_neg_x = I_11 * ev_neg_x + I_12 * ev_neg_y
	I_ev_neg_y = I_12 * ev_neg_x + I_22 * ev_neg_y

	numer_neg = ev_neg_x * II_ev_neg_x + ev_neg_y * II_ev_neg_y
	denom_neg = ev_neg_x * I_ev_neg_x + ev_neg_y * I_ev_neg_y
	denom_neg = sign_floor(denom_neg)
	kappa_neg = numer_neg / denom_neg
	# kappa_neg = jax.lax.cond(numer_neg == 0, lambda: NAN_EPSILON, lambda: numer_neg / denom_neg)

	# Identify min/max curvatures
	abs_kappa_pos = jnp.abs(kappa_pos)
	abs_kappa_neg = jnp.abs(kappa_neg)

	use_pos_as_min = abs_kappa_pos < abs_kappa_neg
	kappa_min = jnp.where(use_pos_as_min, kappa_pos, kappa_neg)
	kappa_max = jnp.where(use_pos_as_min, kappa_neg, kappa_pos)
	v_min = jnp.where(use_pos_as_min, v_pos, v_neg)

	return a00, kappa_min, kappa_max, v_min, normal


@jit
def torus_precompute_masked(
	mask: jnp.ndarray,
	points: jnp.ndarray,
	normals: jnp.ndarray,
	coeffs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	Only compute torii for valid points (where mask is True).

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		points: (max_points, 3) - may contain padding
		normals: (max_points, 3) - may contain padding
		coeffs: (max_points, 6) - may contain padding

	Returns:
		centers: (max_points, 3)
		axes: (max_points, 2)
		major_radii: (max_points,)
		minor_radii: (max_points,)
	"""

	max_points = points.shape[0]

	points, normals, coeffs = _masked_fit_inputs(mask, points, normals, coeffs)

	# Fit torus at a single point.
	@jit
	def solve_single(i):
		p_i = points[i]
		n_i = normals[i]
		a00, kappa_min, kappa_max, v_min, normal = _principal_fit(p_i, n_i, coeffs[i])

		# Geometric sign rule (the osculating torus at every point, saddles included). With the curvatures floored to
		# +-NAN_EPSILON keeping their sign (+NAN_EPSILON at exactly 0) and the signed radii of curvature
		# rho_min = 1/kappa_min, rho_max = 1/kappa_max, the torus center is the center of the parallel circle,
		# Q*(0,0) + rho_min n*, the tube center is Q*(0,0) + rho_max n*, R = |rho_min - rho_max| and the stored minor
		# radius is -rho_max = s r with s = -sgn(kappa_max) (the SDF sign of Eq. 22 is unchanged). The previous rule
		# took the center side from Eq. 22's s, i.e. from kappa_max: at saddles (kappa_min kappa_max < 0) that is the
		# mirror image of the osculating torus, with inside and outside swapped at the point (dg/dn* = -1). Both rules
		# give bit-identical tori wherever kappa_min kappa_max > 0.
		kappa_min_f = jnp.where(jnp.abs(kappa_min) >= NAN_EPSILON, kappa_min, jnp.where(kappa_min < 0, -NAN_EPSILON, NAN_EPSILON))
		kappa_max_f = jnp.where(jnp.abs(kappa_max) >= NAN_EPSILON, kappa_max, jnp.where(kappa_max < 0, -NAN_EPSILON, NAN_EPSILON))
		rho_min = 1.0 / kappa_min_f

		# Compute torus parameters
		r = 1.0 / jnp.abs(kappa_max_f)
		# R = |rho_min - rho_max|, written with a constant sign so that R keeps its gradient where R = 0 (umbilics)
		R = jax.lax.stop_gradient(jnp.sign(kappa_min_f)) * (rho_min - 1.0 / kappa_max_f)

		# Tube sign s = -sgn(kappa_max): the tube is the inside iff its center lies on the inner side
		s = jnp.where(kappa_max_f < 0, 1.0, -1.0)

		# Torus center
		center = p_i + a00 * n_i + rho_min * normal

		# Torus axis (u = normal × v_min, normalized)
		u = jnp.cross(normal, v_min)
		u_norm = jnp.linalg.norm(u)
		u = u / jnp.maximum(u_norm, NAN_EPSILON)

		# Convert to rotation representation
		axis = cartesian_to_rotation_jax(u)  # (2, )

		major_radius = R
		minor_radius = s * r

		return center, axis, major_radius, minor_radius

	# Compute torii for all indices (including invalid ones)
	centers, axes, major_radii, minor_radii = jax.vmap(lambda i: solve_single(i), in_axes=0)(jnp.arange(max_points))
	return centers, axes, major_radii, minor_radii


@jit
def torus_frames_masked(
	mask: jnp.ndarray,
	points: jnp.ndarray,
	normals: jnp.ndarray,
	coeffs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	The tori of torus_precompute_masked in the frame form used by the training blend (torus_distance_gradient_masked):
	the contact point p*, the orthonormal frame (e_x, e_y, e_z) = (v_min / |v_min|, u, n*) with u = n* x v_min / |.| the
	torus axis, and the signed principal curvatures (kappa_min, kappa_max) (not floored). The torus is the same as
	that of (center, axis, R, r) = torus_precompute_masked(...) (geometric sign rule): center p* + n* / kappa_min, axis
	u, R = |1/kappa_min - 1/kappa_max|, r = -1/kappa_max; torus_signed_distance_frame evaluates it without the radii.

	Returns:
		origins: (max_points, 3) contact points p*
		frames: (max_points, 3, 3) rows e_x, e_y (= axis), e_z (= n*)
		kappas: (max_points, 2) (kappa_min, kappa_max)
	"""
	max_points = points.shape[0]
	points, normals, coeffs = _masked_fit_inputs(mask, points, normals, coeffs)

	@jit
	def solve_single(i):
		p_i = points[i]
		n_i = normals[i]
		a00, kappa_min, kappa_max, v_min, normal = _principal_fit(p_i, n_i, coeffs[i])
		u = jnp.cross(normal, v_min)
		u = u / jnp.maximum(jnp.linalg.norm(u), NAN_EPSILON)
		e_x = jnp.cross(u, normal)  # = v_min / |v_min| (v_min is tangent: v_min . n* = 0)
		return p_i + a00 * n_i, jnp.stack([e_x, u, normal]), jnp.stack([kappa_min, kappa_max])

	return jax.vmap(solve_single)(jnp.arange(max_points))


def torus_signed_distance_frame(rel, frame, kappas):
	"""
	Signed distance and its spatial gradient of tori in frame form at the offsets rel = q - p* (broadcasting over the
	leading axes; rel (..., 3), frame (..., 3, 3), kappas (..., 2)).

	With (x, y, z) the frame coordinates of q - p* and k1 = kappa_min, k2 = kappa_max, the explicit torus SDF
	(center p* + n*/k1, axis e_y, R = |1/k1 - 1/k2|, signed minor radius -1/k2) is, after rationalizing both
	sqrt(.) - 1 steps (exact for every sign of k1, k2, and on both sides of the focal points):
		S1 = sqrt((1 - k1 z)^2 + (k1 x)^2),  zeta = (2 z - k1 (x^2 + z^2)) / (S1 + 1)  (the height over the parallel circle),
		S2 = sqrt((1 - k2 zeta)^2 + (k2 y)^2),  g = (2 zeta - k2 (zeta^2 + y^2)) / (S2 + 1),
		grad g = ((1 - k2 zeta) / S2) ((-k1 x) e_x + (1 - k1 z) e_z) / S1 + (-k2 y / S2) e_y.
	There is no 1/kappa: no float32 cancellation of radii ~ 1/kappa against offsets ~ sigma for nearly flat tori
	(g = z at k1 = k2 = 0), and no 1/kappa^2 in the derivatives. S1 = 0 (q on the torus axis) and S2 = 0 (q on the
	tube circle) are the torus' own non-smooth points; there the squares are floored to NAN_EPSILON^2 (a value change
	below float32 resolution elsewhere) so that the derivatives stay finite, as v_safe, d_safe did.
	"""
	x = jnp.sum(rel * frame[..., 0, :], axis=-1)
	y = jnp.sum(rel * frame[..., 1, :], axis=-1)
	z = jnp.sum(rel * frame[..., 2, :], axis=-1)
	k1 = kappas[..., 0]
	k2 = kappas[..., 1]
	a1 = 1.0 - k1 * z
	b1 = k1 * x
	S1 = jnp.sqrt(jnp.maximum(a1 * a1 + b1 * b1, NAN_EPSILON**2))
	zeta = (2.0 * z - k1 * (x * x + z * z)) / (S1 + 1.0)
	a2 = 1.0 - k2 * zeta
	b2 = k2 * y
	S2 = jnp.sqrt(jnp.maximum(a2 * a2 + b2 * b2, NAN_EPSILON**2))
	g = (2.0 * zeta - k2 * (zeta * zeta + y * y)) / (S2 + 1.0)
	cx = (a2 / S2) * (-b1 / S1)  # d g / d x
	cy = -b2 / S2  # d g / d y
	cz = (a2 / S2) * (a1 / S1)  # d g / d z
	grad = cx[..., None] * frame[..., 0, :] + cy[..., None] * frame[..., 1, :] + cz[..., None] * frame[..., 2, :]
	return g, grad


@jit
def torus_signed_distance(q, center, axis, major_radius, minor_radius):
	"""
	Compute the signed distance at q to a single (signed) torus.
	"""
	axis3d = rotation_to_cartesian_jax(axis)

	# Distance from query to torus center
	r_vec = q - center

	# v = |r_vec × axis|
	cross_prod = jnp.cross(r_vec, axis3d)
	v = jnp.linalg.norm(cross_prod, axis=-1)

	# Distance components
	d1 = v - major_radius
	d2 = jnp.sum(r_vec * axis3d, axis=-1)

	# Combined distance to torus surface
	d = jnp.sqrt(d1**2 + d2**2)

	s = jnp.sign(minor_radius)  # (1, 1, max_points)
	g_z = s * (d - jnp.abs(minor_radius))  # (batch, n_queries, max_points)

	return g_z


@partial(jit, static_argnames=['knn'])
def torus_distance_gradient_masked(
	mask: jnp.ndarray,
	queries: jnp.ndarray,
	points: jnp.ndarray,
	origins: jnp.ndarray,
	frames: jnp.ndarray,
	kappas: jnp.ndarray,
	knn: int = 0,
	point_sign: jnp.ndarray = None,
	log_weight: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	Only sum over torii at valid points (where mask is True).

	knn = 0 (default): the blend runs over all valid points, with the shift (and lambda) from the query's 64 nearest
	valid points. knn = K > 0: the blend runs over the K nearest valid points of each query only, with the shift half
	the largest distance in that set and lambda = 64 / shift; this is the rule of the accelerated (k-NN) evaluation
	at inference.

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		queries: (batch_size, n_queries, 3) array of query points
		points: (max_points, 3) - may contain padding
		origins, frames, kappas: the tori in frame form (torus_frames_masked); the per-torus distance is
			torus_signed_distance_frame, the same function as the explicit (center, axis, R, r) torus SDF up to
			rounding, without the float32 cancellation of nearly flat tori and without the (theta, phi) round trip of
			the axis
		point_sign: None or (max_points,) multiplier of each torus' signed distance and gradient (+-1: --aux-heads flip;
			the same as negating the explicit minor radius)
		log_weight: None or (max_points,) added to each torus' blend exponent (log of its confidence: --aux-heads)

	Returns:
		distances: (batch_size, n_queries,) array of signed distances
		gradients: (batch_size, n_queries, 3) array of signed distance gradients
	"""
	batch_size, n_queries, _ = queries.shape

	# Create validity mask for torii without NaNs --- in case there were still NaNs that escaped torus_frames_masked()
	valid_torus_mask = (
		mask
		& jnp.isfinite(origins).all(axis=-1)
		& jnp.isfinite(frames).all(axis=(-2, -1))
		& jnp.isfinite(kappas).all(axis=-1)
	)  # (max_points,)

	# Compute distance and gradient for each query-torus pair

	# Reshape for broadcasting: queries (batch, n_queries, 1, 3)
	q = queries[:, :, None, :]  # (batch, n_queries, 1, 3)

	if knn:
		# Gather the knn nearest valid points of each query: (batch, n_queries, knn[, 3]) arrays replace the
		# (1, 1, max_points[, 3]) broadcasts below; the per-torus formulas are the same.
		d_all = jnp.linalg.norm(q - points[None, None, :, :], axis=-1)
		d_all = jnp.where(valid_torus_mask[None, None, :], d_all, jnp.inf)
		_, nearest = jax.lax.top_k(-jax.lax.stop_gradient(d_all), knn)  # (batch, n_queries, knn)
		p_i = points[nearest]
		rVec = q - p_i  # (batch, n_queries, knn, 3)
		dist_to_point = jnp.linalg.norm(rVec, axis=-1)  # (batch, n_queries, knn)
		mask_i = valid_torus_mask[nearest]
		shifts = 0.5 * jnp.max(jnp.where(mask_i, dist_to_point, 0.0), axis=-1)  # (batch, n_queries)
		o_i = origins[nearest]
		f_i = frames[nearest]
		k_i = kappas[nearest]
		sign_i = None if point_sign is None else point_sign[nearest]
		logw_i = None if log_weight is None else log_weight[nearest]
	else:
		# Broadcast points, centers, etc. for all queries
		# (1, 1, max_points, 3)
		p_i = points[None, None, :, :]

		# Compute query-to-point distances once and reuse for both shift and weight computation.
		# This avoids materializing the (batch, n_queries, max_points) distance tensor twice.
		rVec = q - p_i  # (batch, n_queries, max_points, 3)
		dist_to_point = jnp.linalg.norm(rVec, axis=-1)  # (batch, n_queries, max_points)

		shifts = _compute_shifts_from_distances(dist_to_point, valid_torus_mask)  # (batch, n_queries)
		o_i = origins[None, None, :, :]
		f_i = frames[None, None, :, :, :]
		k_i = kappas[None, None, :, :]
		mask_i = valid_torus_mask[None, None, :]  # (1, 1, max_points)
		sign_i = None if point_sign is None else point_sign[None, None, :]
		logw_i = None if log_weight is None else log_weight[None, None, :]
	lam = compute_lambdas_jax(shifts)  # (batch_size, n_queries)

	# Signed distance and spatial gradient of every query-torus pair (the frame form of the torus SDF)
	g_z, grad_g = torus_signed_distance_frame(q - o_i, f_i, k_i)  # (batch, n_queries, max_points[, 3])
	g_z = jnp.where(mask_i, g_z, 0.0)  # apply mask
	grad_g = jnp.where(mask_i[..., None], grad_g, 0.0)
	if sign_i is not None:
		g_z = sign_i * g_z
		grad_g = sign_i[..., None] * grad_g

	# Compute exponential weights
	rHat = rVec / jnp.maximum(dist_to_point, NAN_EPSILON)[:, :, :, None]

	# Exponential weight
	lam_expanded = lam[:, :, None]  # (batch, n_queries, 1)
	shift_expanded = shifts[:, :, None]  # (batch, n_queries, 1)
	# Log-sum-exp (soft-min) form of the Eq. 25 blend: the exponents are shifted by their per-query maximum over the
	# valid points (a constant, hence stop_gradient), so the largest weight is 1 and sum(weight) >= 1. The blend
	# sum(w g) / sum(w) is unchanged, but the NAN_EPSILON guard below can no longer fire. Before, the shift/lambda came
	# from the 64-NN of the query while the sum ran over all points, so for queries beyond ~0.9x the 64-NN radius every
	# weight underflowed, sum(w) < 1e-10, and the guard returned phi ~ 0 with a meaningless gradient there.
	exponent = -lam_expanded * (dist_to_point - shift_expanded)  # (batch, n_queries, max_points)
	if logw_i is not None:
		exponent = exponent + logw_i
	exponent_max = jnp.max(jnp.where(mask_i, exponent, -jnp.inf), axis=-1, keepdims=True)  # (batch, n_queries, 1)
	exponent_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(exponent_max), exponent_max, 0.0))
	weight = jnp.exp(exponent - exponent_max)  # (batch, n_queries, max_points)

	# Apply mask to weights
	weight = jnp.where(mask_i, weight, 0.0)

	# Accumulate: grad_g_minus_lambda_n_g, g, n
	grad_g_minus_lambda_n_g = jnp.sum(
		weight[:, :, :, None] * (grad_g - lam_expanded[:, :, :, None] * rHat * g_z[:, :, :, None]), axis=2
	)  # (batch, n_queries, 3)

	g = jnp.sum(weight * g_z, axis=2)  # (batch, n_queries)
	n = jnp.sum(weight[:, :, :, None] * rHat, axis=2)  # (batch, n_queries, 3)

	normalization = jnp.sum(weight, axis=2, keepdims=True)  # (batch, n_queries, 1)
	normalization_safe = jnp.maximum(normalization, NAN_EPSILON)

	# Compute final distance and gradient

	A = grad_g_minus_lambda_n_g / normalization_safe  # (batch, n_queries, 3)
	B = lam_expanded * n / normalization_safe  # (batch, n_queries, 3)
	dist = g / normalization_safe[:, :, 0]
	grad = A + dist[:, :, None] * B  # (batch, n_queries, 3)

	return dist, grad


# ========================================================================
# EXPORT FOR VISUALIZATION
# ========================================================================


def generate_torus_mesh(center: np.ndarray, axis: np.ndarray, major_radius: float, minor_radius: float):
	"""
	Generate quad mesh for a single torus.

	Args:
		center: (3,) center position
		axis: (2,) axis in spherical coordinates (theta, phi)
		major_radius: major radius R
		minor_radius: minor radius r (can be negative)

	Returns:
		vertices: (major_res * minor_res, 3)
		faces: (major_res * minor_res, 4) quad indices
	"""
	# Handle degenerate cases
	if np.abs(minor_radius) < 1e-10 or np.abs(major_radius) < 1e-10:
		# Degenerate torus - return empty mesh
		return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32)

	# major_res: number of segments around major circle
	# minor_res: number of segments around minor circle
	max_spacing = 0.1
	min_res = 16  # for small point clouds (a few thousand points), use 16
	max_res = 128
	major_res = min(max(int(major_radius / max_spacing), min_res), max_res)
	minor_res = min(max(int(minor_radius / max_spacing), min_res), max_res)

	# Get torus axis as 3D vector
	axis_3d = rotation_to_cartesian_jax(axis)

	# Create two orthogonal vectors perpendicular to axis
	v1, v2 = orthonormal_basis_with_rotation(axis_3d, 0.0)

	# Generate vertices
	vertices = []

	for i in range(major_res):
		# Angle around major circle
		theta = 2 * np.pi * i / major_res

		# Point on major circle
		major_circle_point = center + major_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)

		# Normal direction at this point on major circle (pointing away from center)
		major_normal = np.cos(theta) * v1 + np.sin(theta) * v2

		# Create local frame for minor circle
		# tangent to major circle
		major_tangent = -np.sin(theta) * v1 + np.cos(theta) * v2

		for j in range(minor_res):
			# Angle around minor circle
			phi = 2 * np.pi * j / minor_res

			# Offset from major circle
			# Note: minor_radius can be negative (changes which side of major circle)
			offset = minor_radius * (np.cos(phi) * major_normal + np.sin(phi) * axis_3d)

			vertex = major_circle_point + offset
			vertices.append(vertex)

	vertices = np.array(vertices)

	# Generate quad faces
	faces = []

	for i in range(major_res):
		for j in range(minor_res):
			v0 = i * minor_res + j
			v1 = ((i + 1) % major_res) * minor_res + j
			v2 = ((i + 1) % major_res) * minor_res + (j + 1) % minor_res
			v3 = i * minor_res + (j + 1) % minor_res

			faces.append([v1, v2, v3, v0])

	faces = np.array(faces, dtype=np.int32)

	return vertices, faces


def export_torus(center: np.ndarray, axis: np.ndarray, major_radius: float, minor_radius: float, save_filepath: str):
	"""
	Export a single torus.

	Args:
		center: (3,)
		axis: (2,)
		major_radius: scalar
		minor_radius: scalar
	"""
	vertices, faces = generate_torus_mesh(center, axis, major_radius, minor_radius)
	# export_OBJ(vertices, faces, filepath=save_filepath)
	export_PLY(vertices, faces, filepath=save_filepath, binary=True)


def hsv_to_rgb(h, s, v):
	"""
	Convert HSV to RGB (matching the GLSL hsv2rgb function).

	Args:
		h, s, v: Hue [0,1], Saturation [0,1], Value [0,1]

	Returns:
		(r, g, b) tuple in [0, 1]
	"""
	c = v * s
	x = c * (1 - abs((h * 6) % 2 - 1))
	m = v - c

	if h < 1 / 6:
		r, g, b = c, x, 0
	elif h < 2 / 6:
		r, g, b = x, c, 0
	elif h < 3 / 6:
		r, g, b = 0, c, x
	elif h < 4 / 6:
		r, g, b = 0, x, c
	elif h < 5 / 6:
		r, g, b = x, 0, c
	else:
		r, g, b = c, 0, x

	return (r + m, g + m, b + m)


def torus_color(i):
	"""
	Compute the color of the i-th torus (matching shader logic).

	Args:
		i: Torus index

	Returns:
		(r, g, b) in [0, 255] as uint8
	"""
	incr = (1.0 + np.sqrt(5.0)) / 20.0
	h = (i * incr) % 1.0
	s = 0.5
	v = 1.0

	r, g, b = hsv_to_rgb(h, s, v)
	return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def export_tori(centers: np.ndarray, axes: np.ndarray, major_radii: float, minor_radii: float, save_filepath: str):
	"""
	Export a collection of torii in a single file.

	Args:
		center: (n_torii, 3)
		axis: (n_torii, 2)
		major_radius: (n_torii, )
		minor_radius: (n_torii, )
	"""
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0

	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		print(f'Generating torus mesh {i}...')
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color for this torus (same color for all vertices of this torus)
		color = torus_color(i)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine all meshes
	print('Combining all torus mehses...')
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		print(f'Warning: No valid torii found, empty mesh exported to {save_filepath}')
		with open(save_filepath, 'w') as f:
			f.write('# Empty mesh (no valid torii)\n')


def _export_tori_single(
	centers: np.ndarray,
	axes: np.ndarray,
	major_radii: np.ndarray,
	minor_radii: np.ndarray,
	save_filepath: str,
	global_index_offset: int = 0,
	verbose: bool = True,
) -> int:
	"""
	Export a collection of torii to a single file.

	Args:
		centers: (n_torii, 3)
		axes: (n_torii, 2)
		major_radii: (n_torii,)
		minor_radii: (n_torii,)
		save_filepath: output filepath
		global_index_offset: offset for global indexing (for consistent colors across chunks)
		verbose: print progress

	Returns:
		Number of valid torii exported
	"""
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0
	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color using global index for consistency across chunks
		global_index = global_index_offset + i
		color = torus_color(global_index)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine and export
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		if verbose:
			print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		if verbose:
			print(f'Warning: No valid torii found, skipping {save_filepath}')

	return n_valid


def export_tori_chunked(
	centers: np.ndarray,
	axes: np.ndarray,
	major_radii: np.ndarray,
	minor_radii: np.ndarray,
	save_filepath: str,
	chunk_size: int = 1_000_000,
	verbose: bool = True,
) -> List[str]:
	"""
	Export a collection of torii in chunks, with each chunk saved to a separate file.

	Args:
		centers: (n_torii, 3) torus centers
		axes: (n_torii, 2) torus axes in spherical coordinates
		major_radii: (n_torii,) torus major radii
		minor_radii: (n_torii,) torus minor radii
		save_filepath: base filepath for output (e.g., "output/torii.ply")
					   chunks will be saved as "output/torii_chunk0.ply", etc.
		chunk_size: number of torii per chunk (default 1M)
		verbose: print progress information

	Returns:
		List of output filepaths that were created
	"""
	n_torii = len(centers)
	print(f'# torii: {n_torii}')

	# Parse base filepath
	if save_filepath.endswith('.ply'):
		base_path = save_filepath[:-4]
	else:
		base_path = save_filepath

	# # If small enough, export in one file
	# if n_torii <= chunk_size:
	# 	output_files = []
	# 	filepath = f"{base_path}.ply"
	# 	n_valid = _export_tori_single(
	# 		centers, axes, major_radii, minor_radii,
	# 		filepath, global_index_offset=0, verbose=verbose
	# 	)
	# 	if n_valid > 0:
	# 		output_files.append(filepath)
	# 	return output_files

	# Process in chunks
	n_chunks = (n_torii + chunk_size - 1) // chunk_size

	if verbose:
		print(f'Exporting {n_torii:,} torii in {n_chunks} chunks of up to {chunk_size:,} each...')

	output_files = []
	total_valid = 0

	for chunk_idx in range(n_chunks):
		start = chunk_idx * chunk_size
		end = min(start + chunk_size, n_torii)

		if verbose:
			progress = (chunk_idx + 1) / n_chunks * 100
			print(f'  Chunk {chunk_idx}/{n_chunks - 1} (torii {start:,}-{end - 1:,}, {progress:.1f}%)...')

		# Extract chunk data
		centers_chunk = centers[start:end]
		axes_chunk = axes[start:end]
		major_chunk = major_radii[start:end]
		minor_chunk = minor_radii[start:end]

		# Export chunk with global index offset for consistent coloring
		filepath = f'{base_path}_chunk{chunk_idx}.ply'
		n_valid = _export_tori_single(
			centers_chunk, axes_chunk, major_chunk, minor_chunk, filepath, global_index_offset=start, verbose=False
		)

		if n_valid > 0:
			output_files.append(filepath)
			total_valid += n_valid
			if verbose:
				print(f'	Exported {n_valid:,} valid torii to {filepath}')
		else:
			if verbose:
				print(f'	No valid torii in chunk {chunk_idx}, skipping file creation')

		# Free memory
		del centers_chunk, axes_chunk, major_chunk, minor_chunk
		gc.collect()

	if verbose:
		print(f'  Done. Exported {total_valid:,}/{n_torii:,} valid torii to {len(output_files)} files.')

	return output_files


# ========================================================================
# NEURAL NETWORK ARCHITECTURE
# ========================================================================


class TransformerBlock(nnx.Module):
	"""
	Transformer block using Flax built-in modules.
	"""

	def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, dropout_rate: float = 0.0, *, rngs: nnx.Rngs):
		self.attention = nnx.MultiHeadAttention(
			num_heads=n_heads,
			in_features=embed_dim,
			qkv_features=embed_dim,
			out_features=embed_dim,
			decode=False,
			rngs=rngs,
		)

		# feed-forward network
		self.ffn = nnx.Sequential(
			nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
			nnx.gelu,
			nnx.Dropout(rate=dropout_rate, rngs=rngs),
			nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
			nnx.Dropout(rate=dropout_rate, rngs=rngs),
		)

		# layer normalization
		self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

	def __call__(self, x: jax.Array) -> jax.Array:
		# Self-attention block
		residual = x
		x = self.norm1(x)
		x = self.attention(x)
		x = self.dropout(x)
		x = residual + x

		# Feed-forward block
		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = residual + x

		return x

	def call_first_token(self, x: jax.Array) -> jax.Array:
		"""
		self(x)[:, :1]: the block's output at token 0 only, [batch, 1, embed_dim]. Token 0 attends to all tokens as
		before (its query against the keys and values of every token); LayerNorm and the FFN act per token. The same
		function as self(x)[:, :1] up to rounding, without computing the 63 outputs that are discarded after the last
		block.
		"""
		h = self.norm1(x)
		h = self.attention(h[:, :1], h, h)
		h = self.dropout(h)
		x = x[:, :1] + h

		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = residual + x

		return x


GEOMETRIC_INPUT_DIM = 13
SECANT_EPS = 0.05  # (in units of the neighbourhood's median distance) regularizes the secant curvatures near the centre


def geometric_input_features(coordinates, local_normals):
	"""
	--input-mlp features of each neighbour from its local coordinates (s, t, h) and local normal (n_s, n_t, n_z), all in
	units of the neighbourhood's median distance: [s, t, h, n_s, n_t, n_z, r, rho, s^2, s t, t^2, 2 h / (rho^2 + eps),
	-(s n_s + t n_t) / (rho^2 + eps)] with r = |(s, t, h)|, rho = |(s, t)|; the last two are the secant normal
	curvature of the neighbour seen from the centre, from its height and from its normal (0 at the centre).
	"""
	s, t, h = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
	ns, nt, nz = local_normals[..., 0], local_normals[..., 1], local_normals[..., 2]
	rho2 = s * s + t * t
	r = jnp.sqrt(rho2 + h * h + 1e-12)
	rho = jnp.sqrt(rho2 + 1e-12)
	inv = 1.0 / (rho2 + SECANT_EPS)
	return jnp.stack(
		[s, t, h, ns, nt, nz, r, rho, s * s, s * t, t * t, 2.0 * h * inv, -(s * ns + t * nt) * inv], axis=-1
	)


WLS_RIDGE = 1e-4


def weighted_jet_fit(geometry, w_pos, w_nrm):
	"""
	--wls: the quadric h = a00 + a10 s + a01 t + a20 s^2 + a11 s t + a02 t^2 of a neighbourhood by weighted least squares
	on the neighbours' heights (weights w_pos) and normals (weights w_nrm; the normal (n_s, n_t, n_z) of the quadric's
	graph satisfies n_s + n_z dh/ds = 0, n_t + n_z dh/dt = 0, linear in the coefficients and without division by n_z),
	with a ridge of WLS_RIDGE times the total weight. Differentiable (a batched 6 x 6 solve).

	geometry: [batch, k, 6] (s, t, h, n_s, n_t, n_z) in units of the median distance; w_pos, w_nrm: [batch, k] >= 0.
	Returns [batch, 6] coefficients (a00, a01, a10, a11, a02, a20) in the same units.
	"""
	s, t, h, ns, nt, nz = [geometry[..., i] for i in range(6)]
	one, zero = jnp.ones_like(s), jnp.zeros_like(s)
	phi = jnp.stack([one, t, s, s * t, t * t, s * s], axis=-1)  # height rows [batch, k, 6]
	rs = nz[..., None] * jnp.stack([zero, zero, one, t, zero, 2 * s], axis=-1)  # dh/ds rows
	rt = nz[..., None] * jnp.stack([zero, one, zero, s, 2 * t, zero], axis=-1)  # dh/dt rows
	wp, wn = w_pos[..., None, None], w_nrm[..., None, None]
	M = jnp.sum(
		wp * phi[..., :, None] * phi[..., None, :] + wn * (rs[..., :, None] * rs[..., None, :] + rt[..., :, None] * rt[..., None, :]),
		axis=1,
	)
	b = jnp.sum(w_pos[..., None] * phi * h[..., None] - w_nrm[..., None] * (rs * ns[..., None] + rt * nt[..., None]), axis=1)
	ridge = WLS_RIDGE * jnp.sum(w_pos + w_nrm, axis=-1)[:, None, None] + 1e-12
	return jnp.linalg.solve(M + ridge * jnp.eye(6, dtype=M.dtype), b[..., None])[..., 0]


class CrossAttentionBlock(nnx.Module):
	"""Pre-norm cross-attention block: queries x attend to context y, then a feed-forward block on x."""

	def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
		self.attention = nnx.MultiHeadAttention(
			num_heads=n_heads, in_features=embed_dim, qkv_features=embed_dim, out_features=embed_dim, decode=False, rngs=rngs
		)
		self.ffn = nnx.Sequential(nnx.Linear(embed_dim, mlp_dim, rngs=rngs), nnx.gelu, nnx.Linear(mlp_dim, embed_dim, rngs=rngs))
		self.norm_q = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.norm_kv = nnx.LayerNorm(embed_dim, rngs=rngs)
		self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)

	def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
		kv = self.norm_kv(y)
		x = x + self.attention(self.norm_q(x), kv, kv)
		return x + self.ffn(self.norm2(x))


class InputFeatures(nnx.Module):
	"""
	Turn a neighborhood (point positions & their normals) to the input features to be fed into the neural network.
	"""

	def __init__(self, output_dim: int, n_freq: int = 0, input_mlp: bool = False, *, rngs: nnx.Rngs):
		# Input features: local coordinates (3), normals in local coordinates (3)
		input_dim = 6

		self.n_freq = n_freq
		if n_freq > 0:
			pos_encoding_dim = 3 * (2 * n_freq + 1)
			input_dim = pos_encoding_dim + 3

		# A linear projection simply to match transformer dimension
		self.mlp = nnx.Sequential(
			nnx.Linear(input_dim, output_dim, rngs=rngs),
			nnx.LayerNorm(output_dim, rngs=rngs),  # LayerNorm seems to really help here
		)
		self.input_mlp = input_mlp
		if input_mlp:
			# --input-mlp: the 13 geometric features of geometric_input_features through a 2-layer GELU MLP
			self.mlp = nnx.Sequential(
				nnx.Linear(GEOMETRIC_INPUT_DIM, output_dim, rngs=rngs),
				nnx.gelu,
				nnx.Linear(output_dim, output_dim, rngs=rngs),
				nnx.LayerNorm(output_dim, rngs=rngs),
			)

		self.key = jax.random.PRNGKey(57)

	def __call__(
		self,
		positions: jax.Array,  # [batch, k, 3]
		normals: jax.Array,  # [batch, k, 3]
		basis=None,
		return_geometry=False,
	) -> jax.Array:
		"""
		Compute geometric features relative to first point (center).

		basis: None (the legacy frame orthonormal_basis_with_rotation(n, 0)) or (s_hat, t_hat), each [batch, 1, 3]: the
			tangent frame of the local coordinates (FundamentalFormPredictor frame options).
		return_geometry: also return the [batch, k, 6] local coordinates and normals (s, t, h, n_s, n_t, n_z) in units of
			the median distance (--wls).

		Returns:
			[batch, k, output_dim] geometric encodings
		"""
		batch_size, k, _ = positions.shape

		norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)  # [batch, k, 1]
		normalized_normals = normals / jnp.maximum(norms, NAN_EPSILON)  # [batch, k, 3]

		# Center point (index 0)
		center_normal = normalized_normals[:, 0:1, :]  # [batch, 1, 3]

		# Regularization: scale neighborhood according to median_j |p_i - p_j|
		diff_vectors = positions - positions[:, 0:1, :]  # [batch, k, 3]
		distances = jnp.sqrt(jnp.sum(diff_vectors * diff_vectors, axis=-1))  # [batch, k]
		median = jnp.median(distances, axis=-1)  # [batch, ]
		# A neighborhood whose points mostly coincide (the padded rows of a padded shape) has median 0, and 0/0 gives NaN
		# features. Even when such rows are masked out of the loss, NaN activations turn the (zero) cotangent of every
		# shared weight into NaN, i.e. the whole gradient of the step. Use scale 1 there instead.
		median = jnp.where(median > 0, median, 1.0)

		# The following block of code is copied from local_coordinates() above, because we also need the orthonormal basis.
		s_hat, t_hat = orthonormal_basis_with_rotation(center_normal, 0.0) if basis is None else basis

		# Encode relative positions in local frame as (s, t, Q(s, t)) tuples. The offsets (p_j - p_0) / median, the same
		# as p_j / median - p_0 / median up to rounding, without the float32 cancellation of clouds far from the origin.
		diff_vectors = diff_vectors / median[:, None, None]  # [batch, k, 3]
		heights = jnp.sum(diff_vectors * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
		S = jnp.sum(diff_vectors * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		T = jnp.sum(diff_vectors * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		coordinates = jnp.concatenate([S, T, heights], axis=-1)  # [batch, k, 3]

		# Encode normals in local frame.
		S = jnp.sum(normalized_normals * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		T = jnp.sum(normalized_normals * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
		N = jnp.sum(normalized_normals * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
		local_normals = jnp.concatenate([S, T, N], axis=-1)  # [batch, k, 3]

		# [local coordinates (3), normals in local coordinates (3)]
		geometric_features = jnp.concatenate([coordinates, local_normals], axis=-1)  # [batch, k, input_dim]
		local_geometry = geometric_features
		if self.input_mlp:
			geometric_features = geometric_input_features(coordinates, local_normals)
		if return_geometry:
			return self.mlp(geometric_features), median, local_geometry

		return self.mlp(geometric_features), median


# Architecture options of FundamentalFormPredictor (train.py flags; saved with the weights). The defaults are the
# authors' network.
FRAMES = ('legacy', 'continuous', 'invariant')
ARCH_DEFAULTS = dict(
	frame='legacy',  # tangent frame of the input features (see continuous_frame, invariant_frames)
	input_mlp=False,  # geometric input features through a 2-layer MLP (geometric_input_features)
	encoder='transformer',  # 'transformer' (the authors' n_layers blocks) or 'perceiver' (latent cross-attention)
	n_latents=8,  # perceiver: number of latent tokens
	n_cross=2,  # perceiver: cross-attention blocks (latents <- neighbour tokens)
	n_self=2,  # perceiver: latent self-attention blocks after each cross-attention
	wls=False,  # coefficients = weighted least-squares jet fit (learned weights) + bounded residual (weighted_jet_fit)
	wls_bound=1.0,  # --wls: bound of the residual (units of the neighbourhood's median distance)
	aux_heads=False,  # two more outputs per point: normal-flip logit and log-confidence of its torus (aux_blend_terms)
)


class FundamentalFormPredictor(nnx.Module):
	def __init__(
		self,
		embed_dim: int = 128,
		n_freq: int = 0,
		n_layers: int = 4,
		n_heads: int = 4,
		mlp_dim: int = 256,
		dropout_rate: float = 0.0,
		*,
		rngs: nnx.Rngs,
		**arch,
	):
		"""
		Args: Parameters for each TransformerBlock.
			n_layers: Number of TransformerBlocks.
			arch: architecture options (ARCH_DEFAULTS; all off by default = the authors' network). They are saved with
				the weights (key 'arch' of a train.py .pkl) and restored by load_saved_model.
		"""
		unknown = set(arch) - set(ARCH_DEFAULTS)
		if unknown:
			raise ValueError(f'unknown architecture options {sorted(unknown)}')
		self.arch = {**ARCH_DEFAULTS, **arch}
		if self.arch['frame'] not in FRAMES:
			raise ValueError(f"frame {self.arch['frame']!r}: expected one of {FRAMES}")
		self.extended = self.arch != ARCH_DEFAULTS

		self.input_encoder = InputFeatures(embed_dim, n_freq=n_freq, input_mlp=self.arch['input_mlp'], rngs=rngs)

		if self.arch['encoder'] == 'perceiver':
			# --encoder perceiver: a pointwise MLP on the k neighbour tokens, n_latents learned latent tokens that
			# cross-attend to them (n_cross times), each cross-attention followed by n_self latent self-attention blocks;
			# readout: the mean of the latents. No k x k attention and no per-token FFN after the first MLP.
			self.blocks = nnx.List([])
			self.latents = nnx.Param(0.02 * jax.random.normal(rngs.params(), (self.arch['n_latents'], embed_dim)))
			self.token_mlp = nnx.Sequential(
				nnx.LayerNorm(embed_dim, rngs=rngs),
				nnx.Linear(embed_dim, embed_dim, rngs=rngs),
				nnx.gelu,
				nnx.Linear(embed_dim, embed_dim, rngs=rngs),
			)
			self.cross_blocks = nnx.List(
				[CrossAttentionBlock(embed_dim, n_heads, mlp_dim, rngs=rngs) for _ in range(self.arch['n_cross'])]
			)
			self.latent_blocks = nnx.List(
				[
					TransformerBlock(embed_dim, n_heads, mlp_dim, dropout_rate, rngs=rngs)
					for _ in range(self.arch['n_cross'] * self.arch['n_self'])
				]
			)
		elif self.arch['encoder'] == 'transformer':
			self.blocks = nnx.List(
				[TransformerBlock(embed_dim, n_heads, mlp_dim, dropout_rate, rngs=rngs) for _ in range(n_layers)]
			)
		else:
			raise ValueError(f"encoder {self.arch['encoder']!r}: expected transformer or perceiver")

		self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

		if self.arch['wls']:
			# --wls: per-neighbour weights (position, normal) of the jet fit, softplus(0.5413) = 1 at initialization
			# (plain least squares); the head below then predicts the bounded residual, zero at initialization.
			self.wls_weights = nnx.Linear(
				embed_dim,
				2,
				kernel_init=nnx.initializers.zeros_init(),
				bias_init=nnx.initializers.constant(jnp.array([0.5413, 0.5413])),
				rngs=rngs,
			)
			if self.arch['encoder'] == 'perceiver':
				# per-token features for the weights: the neighbour tokens attend to the final latents
				self.token_decoder = CrossAttentionBlock(embed_dim, n_heads, mlp_dim, rngs=rngs)

		if self.arch['aux_heads']:
			# --aux-heads: (flip logit, confidence logit) of the point; initially +1 (no flip, tanh' = 0.42) and 0 (all
			# confidences equal, which leaves the normalized blend unchanged)
			self.aux_head = nnx.Linear(
				embed_dim,
				2,
				kernel_init=nnx.initializers.zeros_init(),
				bias_init=nnx.initializers.constant(jnp.array([1.0, 0.0])),
				rngs=rngs,
			)

		# Projection
		# (a00, a01, a10, a11, a02, a20)
		self.mlp = nnx.Sequential(
			nnx.Linear(
				embed_dim,
				6,
				kernel_init=nnx.initializers.zeros_init(),
				# bias_init=nnx.initializers.ones_init(),
				bias_init=nnx.initializers.constant(
					jnp.zeros(6) if self.arch['wls'] else jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5])
				),
				rngs=rngs,
			),
		)

	@nnx.jit
	def __call__(self, positions: jax.Array, normals: jax.Array) -> jax.Array:
		"""
		Args:
			positions, normals: [batch, k, 3] representing a neighborhood of size k; the first point (index 0) is the center

		Returns:
			coefficients: [batch, 6] (a00, a01, a10, a11, a02, a20)
		"""
		if self.extended:
			return self.predict(positions, normals)[0]

		batch_size, k, d = positions.shape

		# input_features: [batch, k, embed_dim]
		# scale: [batch, ]
		input_features, scale = self.input_encoder(positions, normals)
		x = input_features

		# Apply transformer blocks. Only token 0 (the center point) is read out, so the last block computes its output
		# at token 0 only (call_first_token).
		n_blocks = len(self.blocks)
		for i in range(n_blocks - 1):
			x = self.blocks[i](x)
		if n_blocks:
			x = self.blocks[n_blocks - 1].call_first_token(x)  # [batch, 1, embed_dim]

		x = self.final_norm(x)  # [batch, 1, embed_dim]

		# Extract center point
		x = x[:, 0, :]  # [batch, embed_dim]

		# Predict form coefficients
		coeffs = self.mlp(x)  # [batch, 6]

		# Scale back to original space
		coeffs = jnp.stack(
			[
				coeffs[:, 0] * scale,  # a00
				coeffs[:, 1],  # a01
				coeffs[:, 2],  # a10
				coeffs[:, 3] / scale,  # a11
				coeffs[:, 4] / scale,  # a02
				coeffs[:, 5] / scale,  # a20
			],
			axis=-1,
		)

		return coeffs

	def _forward_in_frame(self, positions, normals, s_hat, t_hat):
		"""The network on neighborhoods expressed in the tangent frame (s_hat, t_hat) ([batch, 1, 3] each) -> (the six
		coefficients in that frame, scaled back to world units; the aux logits [batch, 2] or None)."""
		x, scale, geometry = self.input_encoder(positions, normals, basis=(s_hat, t_hat), return_geometry=True)
		if self.arch['encoder'] == 'perceiver':
			tokens = x + self.token_mlp(x)
			lat = jnp.broadcast_to(self.latents[...], (x.shape[0],) + self.latents.shape)
			n_self = self.arch['n_self']
			for i, cross in enumerate(self.cross_blocks):
				lat = cross(lat, tokens)
				for block in self.latent_blocks[i * n_self : (i + 1) * n_self]:
					lat = block(lat)
			lat = self.final_norm(lat)
			x = jnp.mean(lat, axis=1)
			per_token = self.token_decoder(tokens, lat) if self.arch['wls'] else None
		elif self.arch['wls']:
			for block in self.blocks:
				x = block(x)
			per_token = self.final_norm(x)  # the WLS weights need the final features of all tokens
			x = per_token[:, 0, :]
		else:
			# Only token 0 is read out: the last block computes its output at token 0 only (call_first_token)
			n_blocks = len(self.blocks)
			for i in range(n_blocks - 1):
				x = self.blocks[i](x)
			if n_blocks:
				x = self.blocks[n_blocks - 1].call_first_token(x)
			per_token = None
			x = self.final_norm(x)[:, 0, :]
		c = self.mlp(x)
		aux = self.aux_head(x) if self.arch['aux_heads'] else None
		if self.arch['wls']:
			w = jax.nn.softplus(self.wls_weights(per_token))  # [batch, k, 2]
			c = weighted_jet_fit(geometry, w[..., 0], w[..., 1]) + self.arch['wls_bound'] * jnp.tanh(c)
		c = jnp.stack([c[:, 0] * scale, c[:, 1], c[:, 2], c[:, 3] / scale, c[:, 4] / scale, c[:, 5] / scale], axis=-1)
		return c, aux

	@nnx.jit
	def predict(self, positions: jax.Array, normals: jax.Array):
		"""
		The network with its architecture options (self.arch) -> (coefficients [batch, 6] in the legacy frame, aux):
		aux is None, or with aux_heads the [batch, 2] logits (flip, confidence) of aux_blend_terms.
		"""
		n0 = normals[:, 0:1, :]
		n0 = n0 / jnp.maximum(jnp.linalg.norm(n0, axis=-1, keepdims=True), NAN_EPSILON)
		s_leg, t_leg = orthonormal_basis_with_rotation(n0, 0.0)
		if self.arch['frame'] == 'invariant':
			angles, weights = invariant_frames(positions, s_leg, t_leg)  # [batch, F]
			# One frame after another, each under remat (rematerialized in the backward pass): training memory about
			# that of one network pass instead of three, for one more forward pass per frame.
			forward = nnx.remat(lambda m, p, n, s_, t_: m._forward_in_frame(p, n, s_, t_))
			c_sum, aux_sum = 0.0, 0.0
			for f in range(angles.shape[1]):
				co, si = jnp.cos(angles[:, f]), jnp.sin(angles[:, f])
				s_hat = co[:, None, None] * s_leg + si[:, None, None] * t_leg
				t_hat = -si[:, None, None] * s_leg + co[:, None, None] * t_leg
				c, aux = forward(self, positions, normals, s_hat, t_hat)
				c_sum = c_sum + weights[:, f, None] * rotate_coeffs(c, co, si)
				if aux is not None:
					aux_sum = aux_sum + weights[:, f, None] * aux
			return c_sum, (aux_sum if self.arch['aux_heads'] else None)
		if self.arch['frame'] == 'continuous':
			s_hat, t_hat = continuous_frame(n0)
		else:
			s_hat, t_hat = s_leg, t_leg
		c, aux = self._forward_in_frame(positions, normals, s_hat, t_hat)
		# the frame (s_hat, t_hat) is the legacy one rotated by phi: s_hat = cos phi s_leg + sin phi t_leg
		cos_phi = jnp.sum(s_hat * s_leg, axis=-1)[:, 0]
		sin_phi = jnp.sum(s_hat * t_leg, axis=-1)[:, 0]
		return rotate_coeffs(c, cos_phi, sin_phi), aux

	def precompute_coefficients(self, points, normals, k_neighbors, outliers=np.array([])):
		nnx.eval_mode(self)
		# Generate neighborhoods
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)

		# Run forward evaluation.
		predicted_coefficients = self(points[indices], normals[indices])  # (|P|, 6)

		return predicted_coefficients

	def precompute_coefficients_in_chunks(
		self, points, normals, k_neighbors, chunk_size=50000, outliers=np.array([]), return_aux=False
	):
		"""
		Process in chunks to avoid GPU memory issues
		return_aux: also return the (|P|, 2) aux logits (aux_heads models; None otherwise)
		"""
		nnx.eval_mode(self)

		n_points = points.shape[0]
		print(f'Processing {n_points} points in chunks of {chunk_size}...')

		# Generate patches
		t0 = time.time()
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)
		t1 = time.time()
		print(f'  get_neighbors: {t1 - t0:.2f}s')

		# Allocate output array
		all_coefficients = []
		all_aux = []

		# Process in chunks
		n_chunks = (n_points + chunk_size - 1) // chunk_size
		t0 = time.time()

		for chunk_idx in range(n_chunks):
			start_idx = chunk_idx * chunk_size
			end_idx = min(start_idx + chunk_size, n_points)

			# Extract chunk
			chunk_indices = indices[start_idx:end_idx]  # (chunk_size, k)
			chunk_points = points[chunk_indices]  # (chunk_size, k, 3)
			chunk_normals = normals[chunk_indices]  # (chunk_size, k, 3)

			# Forward pass for this chunk
			if return_aux and self.arch['aux_heads']:
				chunk_coeffs, chunk_aux = self.predict(chunk_points, chunk_normals)
				all_aux.append(np.array(chunk_aux))
			else:
				chunk_coeffs = self(chunk_points, chunk_normals)

			# Wait for GPU to finish and transfer to CPU
			chunk_coeffs_cpu = np.array(chunk_coeffs.block_until_ready())
			all_coefficients.append(chunk_coeffs_cpu)

			# Clear JAX cache for this chunk
			del chunk_coeffs

			if (chunk_idx + 1) % 10 == 0:
				print(f'	{chunk_idx + 1}/{n_chunks} chunks processed')

		t1 = time.time()
		print(f'  Coefficient prediction (chunked): {t1 - t0:.2f}s')

		# Concatenate all chunks
		predicted_coefficients = np.concatenate(all_coefficients, axis=0)

		if return_aux:
			return predicted_coefficients, (np.concatenate(all_aux, axis=0) if all_aux else None)
		return predicted_coefficients

	@staticmethod
	def load_saved_model(filepath: str, frame: str = None):
		"""
		Given the filepath where weights are saved, return the model and its parameters.
		frame: None (the saved model's) or a frame option (FRAMES) to evaluate the same weights in, e.g. 'invariant' (the
			frame options do not change the parameters).
		"""
		with open(filepath, 'rb') as f:
			saved_data = pickle.load(f)

		n_freq = saved_data['n_freq'] if 'n_freq' in saved_data else 0
		arch = saved_data.get('arch', {})  # architecture options; absent in the released .pkl (= the defaults)
		if frame is not None:
			arch = {**arch, 'frame': frame}

		loaded_model = FundamentalFormPredictor(
			embed_dim=saved_data['embed_dim'],
			n_freq=n_freq,
			n_heads=saved_data['n_heads'],
			n_layers=saved_data['n_layers'],
			mlp_dim=saved_data['mlp_dim'],
			dropout_rate=saved_data['dropout_rate'],  # not used in eval mode
			rngs=nnx.Rngs(0),  # not used in eval mode
			**arch,
		)

		# Load saved parameters into the model
		nnx.update(loaded_model, saved_data['params'])

		# Put in eval mode
		nnx.eval_mode(loaded_model)

		print(f'Loaded FundamentalFormPredictor model:')
		print(f'  embed_dim: {saved_data["embed_dim"]}')
		print(f'  n_freq: {n_freq}')
		print(f'  n_heads: {saved_data["n_heads"]}')
		print(f'  n_layers: {saved_data["n_layers"]}')
		print(f'  mlp_dim: {saved_data["mlp_dim"]}')
		print(f'  dropout_rate: {saved_data["dropout_rate"]}')
		if arch:
			print(f'  arch: {arch}')

		return loaded_model, saved_data['k_neighbors']
