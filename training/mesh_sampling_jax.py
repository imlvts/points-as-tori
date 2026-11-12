import numpy as np
import time

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from pointsastori.shape_3d import *
from mesh_data import *

EPS = 1e-10


@jit
def closest_point_on_triangle(
	query: jax.Array,  # (3,) or (..., 3)
	v0: jax.Array,  # (3,)
	v1: jax.Array,  # (3,)
	v2: jax.Array,  # (3,)
) -> jax.Array:
	"""
	Find closest point on triangle to query point.

	Args:
		query: Query point(s)
		v0, v1, v2: Triangle vertices

	Returns:
		Closest point on triangle (same shape as query)
	"""
	# Edges
	e0 = v1 - v0
	e1 = v2 - v1
	e2 = v0 - v2

	# Face normal
	n = jnp.cross(e0, -e2)
	n_len = jnp.linalg.norm(n)

	# Handle degenerate triangle
	n_normalized = jnp.where(n_len < EPS, jnp.array([1.0, 0.0, 0.0]), n / n_len)

	# Project query onto plane
	v0_to_query = query - v0
	dist_to_plane = jnp.dot(v0_to_query, n_normalized)
	proj = query - dist_to_plane * n_normalized

	# Check if projection is inside triangle using barycentric sign test
	c0 = jnp.cross(v1 - v0, proj - v0)
	c1 = jnp.cross(v2 - v1, proj - v1)
	c2 = jnp.cross(v0 - v2, proj - v2)

	inside = (
		(jnp.dot(c0, n_normalized) >= 0.0) & (jnp.dot(c1, n_normalized) >= 0.0) & (jnp.dot(c2, n_normalized) >= 0.0)
	)

	# If inside, return projection
	# Otherwise, find closest point on edges

	def closest_on_edge(va: jax.Array, vb: jax.Array) -> Tuple[jax.Array, jax.Array]:
		"""Find closest point on edge and its squared distance."""
		edge = vb - va
		edge_len_sq = jnp.dot(edge, edge)
		t = jnp.dot(query - va, edge) / jnp.maximum(edge_len_sq, EPS)
		t = jnp.clip(t, 0.0, 1.0)
		pt = va + t * edge
		dist_sq = jnp.sum((query - pt) ** 2)
		return pt, dist_sq

	# Check all three edges
	pt0, dist_sq0 = closest_on_edge(v0, v1)
	pt1, dist_sq1 = closest_on_edge(v1, v2)
	pt2, dist_sq2 = closest_on_edge(v2, v0)

	# Find minimum
	min_dist_sq = jnp.minimum(jnp.minimum(dist_sq0, dist_sq1), dist_sq2)

	# Select closest edge point
	edge_closest = jnp.where(dist_sq0 == min_dist_sq, pt0, jnp.where(dist_sq1 == min_dist_sq, pt1, pt2))

	# Return projection if inside, otherwise closest edge point
	return jnp.where(inside, proj, edge_closest)


@jit
def unsigned_distance_to_triangle(
	query: jax.Array,  # (..., 3)
	v0: jax.Array,  # (3,)
	v1: jax.Array,  # (3,)
	v2: jax.Array,  # (3,)
) -> jax.Array:
	"""
	Compute unsigned distance from query point(s) to triangle.

	Args:
		query: Query point(s) of shape (..., 3)
		v0, v1, v2: Triangle vertices

	Returns:
		Unsigned distance(s) of shape (...)
	"""
	closest_pt = closest_point_on_triangle(query, v0, v1, v2)
	return jnp.linalg.norm(query - closest_pt, axis=-1)


@jit
def gwn_single_face(
	query: jax.Array,  # (3,)
	v0: jax.Array,  # (3,)
	v1: jax.Array,  # (3,)
	v2: jax.Array,  # (3,)
) -> jax.Array:
	"""
	Compute solid angle contribution of one triangle to generalized winding number.

	This is a direct JAX port of the GWN computation in shape_3d.cpp.

	Args:
		query: Query point
		v0, v1, v2: Triangle vertices

	Returns:
		Solid angle contribution (scalar)
	"""
	# Vectors from query to vertices
	a_vec = v0 - query
	b_vec = v1 - query
	c_vec = v2 - query

	# Lengths
	a = jnp.linalg.norm(a_vec)
	b = jnp.linalg.norm(b_vec)
	c = jnp.linalg.norm(c_vec)

	# Solid angle formula from C++:
	# numer = -a_z * b_y * c_x + a_y * b_z * c_x + a_z * b_x * c_y
	# - a_x * b_z * c_y - a_y * b_x * c_z + a_x * b_y * c_z
	# This is the determinant det([a, b, c])
	numer = (
		-a_vec[2] * b_vec[1] * c_vec[0]
		+ a_vec[1] * b_vec[2] * c_vec[0]
		+ a_vec[2] * b_vec[0] * c_vec[1]
		- a_vec[0] * b_vec[2] * c_vec[1]
		- a_vec[1] * b_vec[0] * c_vec[2]
		+ a_vec[0] * b_vec[1] * c_vec[2]
	)

	denom = a * b * c + jnp.dot(a_vec, b_vec) * c + jnp.dot(b_vec, c_vec) * a + jnp.dot(c_vec, a_vec) * b

	return 2.0 * jnp.arctan2(numer, denom)


@jit
def compute_face_properties(
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3) - integer indices
	face_mask: jax.Array,  # (max_faces,) - bool mask
) -> Tuple[jax.Array, jax.Array, jax.Array]:
	"""
	Compute face normals, areas, and centroids for all faces.

	Args:
		vertices: Vertex positions (padded)
		faces: Face indices (padded, invalid faces have -1)
		face_mask: Boolean mask indicating valid faces

	Returns:
		normals: (max_faces, 3) - unit normals (zero for invalid faces)
		areas: (max_faces,) - face areas (zero for invalid faces)
		centroids: (max_faces, 3) - face centers (zero for invalid faces)
	"""
	# Extract triangle vertices
	# Handle invalid indices by clamping to 0 (will be masked out anyway)
	safe_faces = jnp.maximum(faces, 0)

	v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	# Centroids
	centroids = (v0 + v1 + v2) / 3.0

	# Area normals (unnormalized)
	area_normals = jnp.cross(v1 - v0, v2 - v0)  # (max_faces, 3)
	double_areas = jnp.linalg.norm(area_normals, axis=1)  # (max_faces,)
	areas = 0.5 * double_areas

	# Normalized normals (handle zero-area faces)
	normals = area_normals / jnp.maximum(double_areas[:, None], EPS)

	# Mask out invalid faces
	normals = jnp.where(face_mask[:, None], normals, 0.0)
	areas = jnp.where(face_mask, areas, 0.0)
	centroids = jnp.where(face_mask[:, None], centroids, 0.0)

	return normals, areas, centroids


@partial(jit, static_argnames=['n'])
def sample_bbox_uniform(
	bbox_min: jax.Array,  # (3,)
	bbox_max: jax.Array,  # (3,)
	n: int,
	key: jax.random.PRNGKey,
) -> jax.Array:
	"""
	Sample points uniformly in a bounding box.

	Args:
		bbox_min: Minimum corner of bbox
		bbox_max: Maximum corner of bbox
		n: Number of samples
		key: Random key

	Returns:
		samples: (n, 3) points in bbox
	"""
	samples = jax.random.uniform(key, (n, 3))
	return bbox_min + samples * (bbox_max - bbox_min)


@partial(jit, static_argnames=['n'])
def sample_zero_set_single_bbox(
	bbox_min: jax.Array,  # (3,)
	bbox_max: jax.Array,  # (3,)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_normals: jax.Array,  # (max_faces, 3)
	face_areas: jax.Array,  # (max_faces,)
	face_mask: jax.Array,  # (max_faces,)
	n: int,
	key: jax.random.PRNGKey,
) -> jax.Array:
	"""
	Sample points uniformly on mesh surface within a bounding box (not exact)

	Strategy:
	1. Test all faces for bbox intersection (brute force)
	2. Sample faces proportionally to area
	3. Sample points uniformly on selected faces

	Args:
		bbox_min, bbox_max: Bounding box
		vertices, faces: Mesh geometry
		face_normals: Pre-computed normals
		face_areas: Pre-computed areas
		face_mask: Valid face mask
		n: Number of samples
		key: Random key

	Returns:
		samples: (n, 3) points on mesh surface within bbox
	"""
	# Extract triangle vertices
	safe_faces = jnp.maximum(faces, 0)
	v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	# Check which faces intersect bbox; approximate by just checking whether triangle bounding box intersects bbox
	tri_bbox_min = jnp.minimum(jnp.minimum(v0, v1), v2)  # (max_faces, 3)
	tri_bbox_max = jnp.maximum(jnp.maximum(v0, v1), v2)

	# AABB-AABB intersection test
	intersects = (
		(tri_bbox_min[:, 0] <= bbox_max[0])
		& (tri_bbox_max[:, 0] >= bbox_min[0])
		& (tri_bbox_min[:, 1] <= bbox_max[1])
		& (tri_bbox_max[:, 1] >= bbox_min[1])
		& (tri_bbox_min[:, 2] <= bbox_max[2])
		& (tri_bbox_max[:, 2] >= bbox_min[2])
	)

	# Combine with face mask
	valid_faces = face_mask & intersects

	# Compute sampling weights (area for valid faces, 0 otherwise)
	weights = jnp.where(valid_faces, face_areas, 0.0)

	# Normalize weights (handle case where no faces intersect)
	total_weight = jnp.sum(weights)
	weights = weights / jnp.maximum(total_weight, EPS)

	# Sample face indices
	key, subkey = jax.random.split(key)
	face_indices = jax.random.choice(subkey, faces.shape[0], shape=(n,), p=weights)

	# Sample triangle
	key, subkey1, subkey2 = jax.random.split(key, 3)
	u = jax.random.uniform(subkey1, (n,))
	v = jax.random.uniform(subkey2, (n,))

	sqrt_u = jnp.sqrt(u)
	alpha = sqrt_u * (1.0 - v)
	beta = v * sqrt_u
	gamma = 1.0 - alpha - beta

	# Vertices for sampled faces
	sampled_v0 = v0[face_indices]  # (n, 3)
	sampled_v1 = v1[face_indices]
	sampled_v2 = v2[face_indices]

	# Compute samples
	samples = alpha[:, None] * sampled_v0 + beta[:, None] * sampled_v1 + gamma[:, None] * sampled_v2

	return samples


@jit
def unsigned_distance_to_mesh(
	queries: jax.Array,  # (n_queries, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
	"""
	Compute unsigned distance from query points to mesh (brute force).

	Args:
		queries: Query points
		vertices: Mesh vertices (padded)
		faces: Mesh faces (padded)
		face_mask: Boolean mask for valid faces

	Returns:
		distances: (n_queries,) unsigned distances to mesh
	"""
	n_queries = queries.shape[0]
	max_faces = faces.shape[0]

	# Triangle vertices
	safe_faces = jnp.maximum(faces, 0)
	v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	# Vectorized distance computation: (n_queries, max_faces)
	# Use vmap to compute distance from each query to each triangle
	def dist_to_all_triangles(q):
		# q: (3,) -> returns (max_faces,)
		return jax.vmap(lambda v0, v1, v2: unsigned_distance_to_triangle(q, v0, v1, v2))(v0, v1, v2)

	distances_all = jax.vmap(dist_to_all_triangles)(queries)  # (n_queries, max_faces)

	# Mask invalid faces with inf
	distances_masked = jnp.where(face_mask[None, :], distances_all, jnp.inf)

	# Take minimum over faces
	return jnp.min(distances_masked, axis=1)  # (n_queries,)


@jit
def gwn_mesh(
	queries: jax.Array,  # (n_queries, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
	"""
	Compute generalized winding number for query points.

	GWN is the sum of solid angles subtended by all faces.

	Args:
		queries: Query points
		vertices: Mesh vertices (padded)
		faces: Mesh faces (padded)
		face_mask: Boolean mask for valid faces

	Returns:
		gwn: (n_queries,) generalized winding numbers (absolute value)
	"""
	# Extract triangle vertices
	safe_faces = jnp.maximum(faces, 0)
	v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	# Compute solid angle contribution for each query-face pair
	def solid_angles_for_query(q):
		# q: (3,) -> returns (max_faces,)
		angles = jax.vmap(lambda v0, v1, v2: gwn_single_face(q, v0, v1, v2))(v0, v1, v2)
		# Mask invalid faces
		return jnp.where(face_mask, angles, 0.0)

	# Sum over all faces for each query
	solid_angles = jax.vmap(solid_angles_for_query)(queries)  # (n_queries, max_faces)
	gwn = jnp.sum(solid_angles, axis=1)  # (n_queries,)

	# Return absolute value for sign test
	return gwn


@jit
def signed_distance_to_mesh(
	queries: jax.Array,  # (n_queries, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
	"""
	Compute signed distance from query points to mesh.

	Uses unsigned distance, and GWN for sign determination.
	Negative inside, positive outside.

	Args:
		queries: Query points
		vertices: Mesh vertices (padded)
		faces: Mesh faces (padded)
		face_mask: Boolean mask for valid faces

	Returns:
		signed_distances: (n_queries,)
	"""
	# Unsigned distance
	unsigned_dist = unsigned_distance_to_mesh(queries, vertices, faces, face_mask)

	# Sign from GWN
	gwn = gwn_mesh(queries, vertices, faces, face_mask)
	sign = jnp.where(gwn > jnp.pi, -1.0, 1.0)

	return sign * unsigned_dist


@partial(jit, static_argnames=['n'])
def sample_zero_set_with_normals(
	bbox_min: jax.Array,  # (3,)
	bbox_max: jax.Array,  # (3,)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_normals: jax.Array,  # (max_faces, 3)
	face_areas: jax.Array,  # (max_faces,)
	face_mask: jax.Array,  # (max_faces,)
	n: int,
	key: jax.random.PRNGKey,
) -> Tuple[jax.Array, jax.Array]:
	"""
	Sample points on mesh surface with their normals.

	Returns:
		samples: (n, 3) points
		normals: (n, 3) surface normals at samples
	"""
	safe_faces = jnp.maximum(faces, 0)
	v0 = vertices[safe_faces[:, 0]]
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	# Check bbox intersection
	tri_bbox_min = jnp.minimum(jnp.minimum(v0, v1), v2)
	tri_bbox_max = jnp.maximum(jnp.maximum(v0, v1), v2)

	intersects = (
		(tri_bbox_min[:, 0] <= bbox_max[0])
		& (tri_bbox_max[:, 0] >= bbox_min[0])
		& (tri_bbox_min[:, 1] <= bbox_max[1])
		& (tri_bbox_max[:, 1] >= bbox_min[1])
		& (tri_bbox_min[:, 2] <= bbox_max[2])
		& (tri_bbox_max[:, 2] >= bbox_min[2])
	)

	valid_faces = face_mask & intersects
	weights = jnp.where(valid_faces, face_areas, 0.0)
	total_weight = jnp.sum(weights)
	weights = weights / jnp.maximum(total_weight, EPS)

	# Sample faces
	key, subkey = jax.random.split(key)
	face_indices = jax.random.choice(subkey, faces.shape[0], shape=(n,), p=weights)

	# Sample triangle
	key, subkey1, subkey2 = jax.random.split(key, 3)
	u = jax.random.uniform(subkey1, (n,))
	v = jax.random.uniform(subkey2, (n,))

	sqrt_u = jnp.sqrt(u)
	alpha = sqrt_u * (1.0 - v)
	beta = v * sqrt_u
	gamma = 1.0 - alpha - beta

	# Get vertices and normals for sampled faces
	sampled_v0 = v0[face_indices]
	sampled_v1 = v1[face_indices]
	sampled_v2 = v2[face_indices]
	sampled_normals = face_normals[face_indices]

	# Compute samples
	samples = alpha[:, None] * sampled_v0 + beta[:, None] * sampled_v1 + gamma[:, None] * sampled_v2

	return samples, sampled_normals


@partial(jit, static_argnames=['n'])
def sample_narrow_band_single_bbox(
	bbox_min: jax.Array,
	bbox_max: jax.Array,
	vertices: jax.Array,
	faces: jax.Array,
	face_normals: jax.Array,
	face_areas: jax.Array,
	face_mask: jax.Array,
	offset: float,
	n: int,
	key: jax.random.PRNGKey,
) -> jax.Array:
	"""
	Sample narrow band around a mesh (approximate)
	"""
	# Sample zero set with normals
	key, subkey = jax.random.split(key)
	zero_samples, normals = sample_zero_set_with_normals(
		bbox_min, bbox_max, vertices, faces, face_normals, face_areas, face_mask, n, subkey
	)

	# Offset along normals
	key, subkey = jax.random.split(key)
	offsets = jax.random.uniform(subkey, (n,), minval=-offset, maxval=offset)

	samples = zero_samples + offsets[:, None] * normals

	return samples


def _pick_accepted(accepted: jax.Array, key: jax.random.PRNGKey, n: int) -> Tuple[jax.Array, jax.Array]:
	"""n indices of accepted entries in random order (cycling if there are fewer than n), and the number accepted."""
	r = jax.random.uniform(key, accepted.shape)
	order = jnp.argsort(jnp.where(accepted, r, 2.0 + r))
	n_accepted = jnp.sum(accepted)
	return order[jnp.arange(n) % jnp.maximum(n_accepted, 1)], n_accepted


@partial(jit, static_argnames=['n', 'n_candidates'])
def sample_zero_set_and_band_local(
	bbox_min: jax.Array,  # (3,) regularized neighborhood box
	bbox_max: jax.Array,  # (3,)
	cube_min: jax.Array,  # (3,) extended cube (the cube-class box)
	cube_max: jax.Array,  # (3,)
	neighborhood_points: jax.Array,  # (k, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_normals: jax.Array,  # (max_faces, 3)
	face_areas: jax.Array,  # (max_faces,)
	face_mask: jax.Array,  # (max_faces,)
	offset: float,
	n: int,
	key: jax.random.PRNGKey,
	n_candidates: int = 4096,
) -> Tuple[jax.Array, jax.Array]:
	"""
	Zero-set and narrow-band samples that stay inside the neighborhood.

	Candidates are drawn area-uniformly on the valid faces whose bounding box meets the regularized box (the same
	candidate faces as before), and a zero-set candidate is kept only if it lies inside that box: rejection sampling,
	i.e. uniform samples of (mesh ∩ box). Sampling the whole intersecting faces instead put most zero-set and band
	queries of CAD meshes with large faces far outside the neighborhood. Band samples are kept zero-set candidates
	offset by U(-offset, offset) along the face normal, kept only if inside the cube box.

	If fewer than n candidates are accepted, accepted ones are reused; if none are (degenerate box), the neighborhood
	points themselves (which lie on the surface) are used.

	Returns:
		zero_samples: (n, 3)
		band_samples: (n, 3)
	"""
	safe_faces = jnp.maximum(faces, 0)
	v0 = vertices[safe_faces[:, 0]]
	v1 = vertices[safe_faces[:, 1]]
	v2 = vertices[safe_faces[:, 2]]

	tri_bbox_min = jnp.minimum(jnp.minimum(v0, v1), v2)
	tri_bbox_max = jnp.maximum(jnp.maximum(v0, v1), v2)
	intersects = jnp.all(tri_bbox_min <= bbox_max, axis=-1) & jnp.all(tri_bbox_max >= bbox_min, axis=-1)
	weights = jnp.where(face_mask & intersects, face_areas, 0.0)
	total_weight = jnp.sum(weights)

	key_face, key_u, key_v, key_zero, key_offset, key_band = jax.random.split(key, 6)
	face_indices = jax.random.choice(
		key_face, faces.shape[0], shape=(n_candidates,), p=weights / jnp.maximum(total_weight, EPS)
	)
	u = jax.random.uniform(key_u, (n_candidates,))
	v = jax.random.uniform(key_v, (n_candidates,))
	sqrt_u = jnp.sqrt(u)
	alpha = sqrt_u * (1.0 - v)
	beta = v * sqrt_u
	gamma = 1.0 - alpha - beta
	candidates = alpha[:, None] * v0[face_indices] + beta[:, None] * v1[face_indices] + gamma[:, None] * v2[face_indices]

	in_box = jnp.all((candidates >= bbox_min) & (candidates <= bbox_max), axis=-1) & (total_weight > 0)
	offsets = jax.random.uniform(key_offset, (n_candidates,), minval=-offset, maxval=offset)
	band = candidates + offsets[:, None] * face_normals[face_indices]
	in_cube = in_box & jnp.all((band >= cube_min) & (band <= cube_max), axis=-1)

	zero_idx, n_zero = _pick_accepted(in_box, key_zero, n)
	band_idx, n_band = _pick_accepted(in_cube, key_band, n)
	fallback = neighborhood_points[jnp.arange(n) % neighborhood_points.shape[0]]
	zero_samples = jnp.where(n_zero > 0, candidates[zero_idx], fallback)
	band_samples = jnp.where(n_band > 0, band[band_idx], fallback)
	return zero_samples, band_samples


@partial(jit, static_argnames=['n_queries', 'n_queries_per_type'])
def generate_queries_for_neighborhood(
	bbox_min: jax.Array,  # (3,)
	bbox_max: jax.Array,  # (3,)
	neighborhood_points: jax.Array,  # (k, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_normals: jax.Array,  # (max_faces, 3)
	face_areas: jax.Array,  # (max_faces,)
	face_mask: jax.Array,  # (max_faces,)
	key: jax.random.PRNGKey,
	n_queries: int = 120,
	n_queries_per_type: int = 40,  # n_queries // 3
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
) -> jax.Array:
	"""
	Generate query points for a single neighborhood.

	Samples:
	- n_queries_per_type from the zero set inside the (regularized) neighborhood box
	- n_queries_per_type from the narrow band, inside the extended cube
	- n_queries_per_type from the extended cube
	(see sample_zero_set_and_band_local)

	Args:
		bbox_min, bbox_max: Neighborhood bounding box
		neighborhood_points: The neighborhood's points (fallback samples for degenerate boxes)
		vertices, faces: Mesh geometry
		face_normals, face_areas: Pre-computed face properties
		face_mask: Valid face mask
		key: Random key
		n_queries: Total queries (should be divisible by 3)
		n_queries_per_type: Queries per type (n_queries // 3)
		narrow_band_offset: Half-width of narrow band
		bbox_extension: Multiplicative extension factor

	Returns:
		queries: (n_queries, 3) query points
	"""
	# Regularize bbox
	bbox_diag = bbox_max - bbox_min
	bbox_extents = jnp.abs(bbox_diag)
	bbox_center = 0.5 * (bbox_max + bbox_min)
	bbox_extents = jnp.maximum(bbox_extents, 0.1)  # Min extent
	bbox_min_reg = bbox_center - 0.5 * bbox_extents
	bbox_max_reg = bbox_center + 0.5 * bbox_extents

	# Split key
	key, key_zero, key_narrow, key_bbox = jax.random.split(key, 4)

	# Extended bbox (cube based on max extent): the cube class, and the box the band samples must stay in
	sidelength = jnp.maximum(jnp.max(jnp.abs(bbox_diag)), 0.01)
	radius = 0.5 * jnp.sqrt(3.0) * sidelength
	diag = jnp.ones(3)
	bbox_min_ext = bbox_center - radius * diag * bbox_extension
	bbox_max_ext = bbox_center + radius * diag * bbox_extension

	# Zero set and narrow band samples
	zero_samples, narrow_samples = sample_zero_set_and_band_local(
		bbox_min_reg,
		bbox_max_reg,
		bbox_min_ext,
		bbox_max_ext,
		neighborhood_points,
		vertices,
		faces,
		face_normals,
		face_areas,
		face_mask,
		narrow_band_offset,
		n_queries_per_type,
		key_zero,
	)

	bbox_samples = sample_bbox_uniform(bbox_min_ext, bbox_max_ext, n_queries_per_type, key_bbox)

	# Concatenate all samples
	queries = jnp.concatenate([zero_samples, narrow_samples, bbox_samples], axis=0)

	return queries


@partial(jit, static_argnames=['n_queries', 'use_unsigned_distance'])
def generate_neighborhood_features(
	neighbor_positions: jax.Array,  # (n_neighborhoods, k, 3)
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_mask: jax.Array,  # (max_faces,)
	key: jax.random.PRNGKey,
	n_queries: int = 120,
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
	use_unsigned_distance: bool = False,
) -> Tuple[jax.Array, jax.Array]:
	"""
	Generate query points and ground-truth distances for the given neighborhoods (point positions of each).

	Args:
		neighbor_positions: The points of each neighborhood
		vertices: Mesh vertices (padded to max_vertices)
		faces: Mesh faces (padded to max_faces, invalid = -1)
		face_mask: Boolean mask for valid faces
		key: Random key for sampling
		n_queries: Total queries per neighborhood (must be divisible by 3)
		narrow_band_offset: Half-width of narrow band
		bbox_extension: Extension factor

	Returns:
		queries: (n_neighborhoods, n_queries, 3) query points per neighborhood
		distances: (n_neighborhoods, n_queries) ground-truth signed distances
	"""
	n_points = neighbor_positions.shape[0]
	n_queries_per_type = n_queries // 3

	# Pre-compute face properties (once for entire shape)
	face_normals, face_areas, _ = compute_face_properties(vertices, faces, face_mask)

	# Compute bboxes for all neighborhoods
	bbox_mins = jnp.min(neighbor_positions, axis=1)  # (n_points, 3)
	bbox_maxs = jnp.max(neighbor_positions, axis=1)

	# Generate keys for each neighborhood
	keys = jax.random.split(key, n_points)

	# Vectorized query generation
	def gen_queries_for_point(i):
		return generate_queries_for_neighborhood(
			bbox_mins[i],
			bbox_maxs[i],
			neighbor_positions[i],
			vertices,
			faces,
			face_normals,
			face_areas,
			face_mask,
			keys[i],
			n_queries,
			n_queries_per_type,
			narrow_band_offset,
			bbox_extension,
		)

	queries = jax.vmap(gen_queries_for_point)(jnp.arange(n_points))  # (n_points, n_queries, 3)

	# Compute signed distances for all queries
	def compute_distances_for_point(point_queries):
		if use_unsigned_distance:
			return unsigned_distance_to_mesh(point_queries, vertices, faces, face_mask)
		return signed_distance_to_mesh(point_queries, vertices, faces, face_mask)

	distances = jax.vmap(compute_distances_for_point)(queries)  # (n_points, n_queries)

	return queries, distances


@partial(jit, static_argnames=['n_queries', 'use_unsigned_distance'])
def generate_shape_features(
	positions: jax.Array,  # (n_points, 3)
	neighbors: jax.Array,  # (n_points, k) - neighbor indices
	vertices: jax.Array,  # (max_vertices, 3)
	faces: jax.Array,  # (max_faces, 3)
	face_mask: jax.Array,  # (max_faces,)
	key: jax.random.PRNGKey,
	n_queries: int = 120,
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
	use_unsigned_distance: bool = False,
) -> Tuple[jax.Array, jax.Array]:
	"""
	Generate query points and ground-truth distances for all neighborhoods in a shape
	(generate_neighborhood_features of positions[neighbors]).

	Returns:
		queries: (n_points, n_queries, 3) query points per neighborhood
		distances: (n_points, n_queries) ground-truth signed distances
	"""
	return generate_neighborhood_features(
		positions[neighbors],
		vertices,
		faces,
		face_mask,
		key,
		n_queries=n_queries,
		narrow_band_offset=narrow_band_offset,
		bbox_extension=bbox_extension,
		use_unsigned_distance=use_unsigned_distance,
	)
