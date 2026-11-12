import argparse
import concurrent.futures

import matplotlib.pyplot as plt

from pointsastori.shape_3d import *
from pointsastori.network import *
from mesh_data import *
from mesh_sampling_jax import *
import augment


def sanitize_neighbors(neighbors):
	"""Replace the -1 entries of a (..., max_points, k) neighbor array by the index of their own row."""
	rows = np.arange(neighbors.shape[-2], dtype=neighbors.dtype)[:, None]
	return np.where(neighbors < 0, rows, neighbors)


def normalize_normals(normals):
	"""Scale every nonzero normal of a (..., 3) array to unit length."""
	norms = np.linalg.norm(normals, axis=-1, keepdims=True)
	return np.where(norms > 0, normals / np.where(norms > 0, norms, 1.0), normals)


def load_mesh_training_data(filepath):
	"""
	Load training data with mesh information from NPZ file.

	Args:
		filepath: Path to NPZ file

	Returns:
		Tuple of NumPy arrays:
			all_positions: (n_shapes, max_points, 3)
			all_normals: (n_shapes, max_points, 3)
			all_neighbors: (n_shapes, max_points, k_neighbors)
			all_masks: (n_shapes, max_points)
			all_mesh_vertices: (n_shapes, max_vertices, 3)
			all_mesh_faces: (n_shapes, max_faces, 3)
			all_face_masks: (n_shapes, max_faces)
	"""
	data = np.load(filepath)

	all_positions = data['positions']
	all_normals = data['normals']
	all_neighbors = data['neighbors']
	all_masks = data['masks']
	all_mesh_vertices = data['mesh_vertices']
	all_mesh_faces = data['mesh_faces']
	all_face_masks = data['face_masks']

	# Padded rows (and the missing neighbors of shapes with fewer than k points) are stored as -1, which would gather
	# the last, zero-padded point. Point them at the row itself instead.
	all_neighbors = sanitize_neighbors(all_neighbors)

	# The normals of pretraining_blob.npz and of the validation files have length 2, the others length 1. The torus
	# fit assumes unit normals (with length L its contact point moves by (L - 1)/|kappa_min| along the normal, which
	# makes over-curvature free during pretraining). Normalize on load; padded (zero) normals stay zero.
	all_normals = normalize_normals(all_normals)

	print(f'Loaded mesh training data from {filepath}')
	print(f'  Shapes: {len(all_positions)} shapes')
	print(f'  Max points per shape: {all_positions.shape[1]}')
	print(f'  k_neighbors: {all_neighbors.shape[2]}')
	print(f'  Max vertices: {all_mesh_vertices.shape[1]}')
	print(f'  Max faces: {all_mesh_faces.shape[1]}')

	return (
		all_positions,
		all_normals,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)


# ========================================================================
# TRAINING
# ========================================================================


@partial(jit, static_argnames=['knn_blend', 'eikonal_weight'])
def SDF_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
	knn_blend=0,
	eikonal_weight=1.0,
):
	"""
	Compute a loss based on the end-to-end SDF process.
	Returns loss that is averaged over all neighborhoods in the batch.

	Args:
		mask: (max_points, ) boolean-valued array indicating which points are actually valid and belong to the input point cloud
		batch_mask: (batch_size, ) boolean-valued array indicating which query points in the batch are valid
		positions, normals: (max_points, 2) the entire input point cloud of a single shape
		coefficients: (max_points, 6) all point coefficients
		queries: (batch_size, n_queries, 2)
		true_distances: (batch_size, n_queries,)
		knn_blend: blend over the knn_blend nearest points of each query (0: all points; see torus_distance_gradient_masked)
		eikonal_weight: weight of the eikonality term (the distance term has weight 1)
	"""

	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii, knn=knn_blend
	)

	n_queries_per_neighborhood = queries.shape[1]
	num_valid = jnp.sum(batch_mask)  # number of valid points in the neighborhood
	distance_error = jnp.abs((predicted_distances - true_distances))  # (batch_size, n_queries, )
	distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[:, None]
	distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)  # (batch_size,)
	num_valid_distance_queries = jnp.sum(distance_valid_mask)

	eikonality_error = jnp.abs(
		1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
	)  # (batch_size, n_queries,)
	eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[:, None]  # (batch_size, n_queries)
	eikonality_error_masked = jnp.sum(
		jnp.where(eikonality_valid_mask, eikonality_error, 0.0), axis=-1
	)  # (batch_size, )
	num_valid_eikonality_queries = jnp.sum(eikonality_valid_mask)

	# Take mean only over valid points in the neighborhood.
	distance_loss = jax.lax.cond(
		num_valid_distance_queries > 0, lambda: jnp.sum(distance_error_masked) / num_valid_distance_queries, lambda: 0.0
	)
	eikonality_loss = jax.lax.cond(
		num_valid_eikonality_queries > 0,
		lambda: jnp.sum(eikonality_error_masked) / num_valid_eikonality_queries,
		lambda: 0.0,
	)

	loss = distance_loss + eikonal_weight * eikonality_loss

	# Return individual components for logging
	loss_dict = {
		'distance_loss': distance_loss,
		'eikonality_loss': eikonality_loss,
	}

	return loss, loss_dict


@partial(jit, static_argnames=['knn_blend'])
def validation_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
	knn_blend=0,
):
	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii, knn=knn_blend
	)

	n_queries_per_neighborhood = queries.shape[1]
	num_valid = jnp.sum(batch_mask)  # number of valid points in the neighborhood
	distance_error = jnp.abs((predicted_distances - true_distances))  # (batch_size, n_queries, )
	distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[:, None]
	distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)  # (batch_size,)
	num_valid_distance_queries = jnp.sum(distance_valid_mask)

	eikonality_error = jnp.abs(
		1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
	)  # (batch_size, n_queries,)
	eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[:, None]  # (batch_size, n_queries)
	eikonality_error_masked = jnp.sum(
		jnp.where(eikonality_valid_mask, eikonality_error, 0.0), axis=-1
	)  # (batch_size, )
	num_valid_eikonality_queries = jnp.sum(eikonality_valid_mask)

	# Take mean only over valid points in the neighborhood.
	distance_loss = jax.lax.cond(
		num_valid_distance_queries > 0, lambda: jnp.sum(distance_error_masked) / num_valid_distance_queries, lambda: 0.0
	)
	eikonality_loss = jax.lax.cond(
		num_valid_eikonality_queries > 0,
		lambda: jnp.sum(eikonality_error_masked) / num_valid_eikonality_queries,
		lambda: 0.0,
	)

	return distance_loss, eikonality_loss


def FundamentalFormPredictorTrainer(
	embed_dim=128,
	n_freq=4,
	n_layers=4,
	n_heads=4,
	mlp_dim=256,
	dropout_rate=0.0,
	rngs=nnx.Rngs(0),
	learning_rate=1e-3,
	weight_decay=0.0,
	warmup_steps=1000,
	total_steps=100000,
	seed=42,
	knn_blend=0,
	clip=0.0,
	eikonal_weight=1.0,
):
	"""
	Set up network (model) with the given parameters.
	eikonal_weight: weight of the eikonality term of the loss
	clip: clip the gradient to this global norm before the AdamW update (0: no clipping)

	Returns: (model, optimizer, train_step) where train_step() is a function that trains the model for a single step.
	"""
	model = FundamentalFormPredictor(
		embed_dim=embed_dim,
		n_freq=n_freq,
		n_layers=n_layers,
		n_heads=n_heads,
		mlp_dim=mlp_dim,
		dropout_rate=dropout_rate,
		rngs=rngs,
	)

	# Learning rate schedule
	schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,  # Start from 0
		peak_value=learning_rate,  # Warmup to this
		warmup_steps=warmup_steps,  # Linear warmup
		decay_steps=total_steps,  # Cosine decay over this many steps
		end_value=learning_rate * 0.01,
	)

	# Optimizer
	optimizer = nnx.Optimizer(
		model,
		optax.chain(
			*([optax.clip_by_global_norm(clip)] if clip > 0 else []),
			optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
		),
		wrt=nnx.Param,
	)

	@nnx.jit
	def train_step(
		model: FundamentalFormPredictor,
		optimizer: nnx.Optimizer,
		mask,
		batch_mask,
		positions,
		normals,
		neighbors,
		queries,
		true_distances,
	):
		"""
		Single training step on a single batch (neighborhood) of one shape.

		Args:
			mask: (max_points,)
			batch_mask: (batch_size, )
			positions, normals: (|P|, 3)
			neighbors: (|P|, k)
			queries: (batch_size * n_queries, 3)
			true_distances: (batch_size * n_queries)

		Returns:
			Loss averaged over batch.
		"""

		def loss_fn(model):
			coefficients = model(positions[neighbors], normals[neighbors])

			loss, loss_dict = SDF_loss(
				mask,
				batch_mask,
				positions,
				normals,
				coefficients,
				queries,
				true_distances,
				knn_blend=knn_blend,
				eikonal_weight=eikonal_weight,
			)

			return loss, loss_dict

		(batch_loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

		def zero_out_nans(x):
			if x is None:
				return x
			return jnp.where(jnp.isnan(x), 0.0, x)

		# Apply to all gradients in the tree
		grads = jax.tree_util.tree_map(zero_out_nans, grads)

		# Update with this batch's gradients
		optimizer.update(model, grads)

		return batch_loss, loss_dict

	return model, optimizer, train_step


# ========================================================================
# MULTI-SHAPE PATCH BATCHES (--multishape)
# ========================================================================


def make_patch_step(knn_blend=0, eikonal_weight=1.0):
	"""
	Train step on S patches (augment.make_patch), one per shape: the network runs on the patch unions, and SDF_loss is
	evaluated per patch (tori and blend restricted to the union) on the queries of the patch centres; the loss is the
	mean over the patches.
	"""

	@nnx.jit
	def patch_step(model, optimizer, UP, UN, Upos, Unrm, Umask, queries, true_distances):
		"""
		UP, UN: (S, U, k, 3) neighborhoods of the union points; Upos, Unrm: (S, U, 3); Umask: (S, U);
		queries: (S, C, n_queries, 3); true_distances: (S, C, n_queries)
		"""
		S, U, K, _ = UP.shape
		batch_mask = jnp.ones(queries.shape[1], bool)

		def loss_fn(model):
			coefficients = model(UP.reshape(S * U, K, 3), UN.reshape(S * U, K, 3)).reshape(S, U, 6)
			_, loss_dict = jax.vmap(
				lambda m, p, n, c, q, d: SDF_loss(
					m, batch_mask, p, n, c, q, d, knn_blend=knn_blend, eikonal_weight=eikonal_weight
				)
			)(Umask, Upos, Unrm, coefficients, queries, true_distances)
			distance_loss = jnp.mean(loss_dict['distance_loss'])
			eikonality_loss = jnp.mean(loss_dict['eikonality_loss'])
			loss = distance_loss + eikonal_weight * eikonality_loss
			return loss, {'distance_loss': distance_loss, 'eikonality_loss': eikonality_loss}

		(loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
		grads = jax.tree_util.tree_map(lambda x: x if x is None else jnp.where(jnp.isnan(x), 0.0, x), grads)
		optimizer.update(model, grads)
		return loss, loss_dict

	return patch_step


def TrainFundamentalFormPredictor(
	training_data_filenames,
	num_epochs,
	learning_rate,
	batch_size,
	warmup_steps,
	n_epochs_to_save,
	n_shapes_to_save,
	load_model_path=None,
	final_save_filepath=None,
	seed=42,
	models_dir=MODELS_DIR,
	validation_file='datasets/ABC17_validation_3.npz',
	max_steps=0,
	log_every=0,
	knn_blend=0,
	min_points=0,
	schedule='authors',
	clip=0.0,
	weight_decay=0.0,
	eikonal_weight=1.0,
	multishape=0,
	patch_centres=32,
	union_size=256,
	save_every=10000,
	resample='',
	resample_match='',
	rotate=False,
	noise=0.0,
	quantize=0.0,
	mix='',
):
	"""
	Args:
		load_model_path: Optional path to a saved model file (.pkl) to use as warm start
		seed: seed of the network initialization, the shape order and the query sampling
		models_dir: directory of the saved models
		validation_file: validation data (skipped if the file does not exist)
		max_steps: stop after this many optimizer steps (0: run all epochs)
		log_every: print the mean loss of the last log_every steps (0: off)
		knn_blend: train (and validate) with the blend over the K nearest points of each query (0: all points)
		min_points: drop training shapes with fewer valid points (0: keep all)
		schedule: 'authors' (linear warmup, then cosine decay over 1e8 steps, i.e. constant in practice) or 'cosine'
			(linear warmup, then cosine decay to 1 % of the learning rate at max_steps)
		clip: clip the gradient to this global norm (0: off)
		weight_decay: AdamW weight decay
		eikonal_weight: weight of the eikonality term of the loss (the distance term has weight 1)
		multishape: S > 0 trains on multi-shape patch batches instead of the authors' loop: every step takes one patch
			of patch_centres clustered query centres from each of S shapes (augment.make_patch; union of their k-NN padded
			to union_size), for max_steps steps; checkpoint and validation every save_every steps. 0: the authors'
			loop (every step trains on batch_size query rows of one shape).
		resample: 'LO:HI' with multishape: every visit of a shape draws fresh points on its ground-truth mesh, their count
			uniform among the powers of two in [LO, HI], and recomputes the k-NN ('': the stored points)
		resample_match: resample only shapes whose data file path contains this string ('': all)
		rotate: apply a random rotation to every shape visit (points, normals and mesh; queries follow the mesh)
		noise: with multishape: Gaussian noise of standard deviation noise * sigma on the input points of every visit
			(sigma: the shape's median k-NN distance; normals and the ground truth stay exact; 0: off)
		quantize: with multishape: round the input points' coordinate along a random direction (one per visit) to
			multiples of quantize * sigma, as a depth scanner does (0: off)
		mix: 'analytic:FRAC:PATH' with multishape: each patch slot is, with probability FRAC, a patch of an analytic scene
			of PATH (gen_analytic.py) with exact-SDF queries (analytic_sdf.analytic_queries) instead of a data shape;
			resample and rotate apply to the analytic scenes too ('': off)
	"""
	LABEL_WIDTH = 18  # Width for "distance loss = ", etc.
	VALUE_WIDTH = 12  # Width for the numeric values

	print(jax.default_backend())
	print(jax.devices())

	key = jax.random.PRNGKey(seed)
	rngs = nnx.Rngs(seed)

	# NOTE: Could probably train on much larger meshes and point clouds using chunked versions of torus_precompute_masked() and precompute_coefficients(), to avoid GPU memory constraints (at the cost of increasing compute time).
	# The current (naive) versions of these functions are called on all points at once (usually in a vectorized manner).

	# Architecture parameters
	embed_dim = 128
	n_freq = 0
	n_layers = 8
	n_heads = 8
	mlp_dim = embed_dim * 4

	# Sampling params
	n_queries_per_neighborhood = 120
	narrow_band_offset = 0.2
	bbox_extension = 1.0

	# Learning parameters
	dropout_rate = 0.0
	if schedule == 'authors':
		total_steps = 1e8
	elif schedule == 'cosine':
		if not max_steps:
			raise ValueError("schedule 'cosine' needs the number of steps of the phase (max_steps)")
		total_steps = max_steps
	else:
		raise ValueError(f'unknown schedule {schedule!r}')

	model, optimizer, train_step = FundamentalFormPredictorTrainer(
		embed_dim=embed_dim,
		n_freq=n_freq,
		n_layers=n_layers,
		n_heads=n_heads,
		mlp_dim=mlp_dim,
		dropout_rate=dropout_rate,
		rngs=rngs,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		warmup_steps=warmup_steps,
		total_steps=total_steps,
		seed=seed,
		knn_blend=knn_blend,
		clip=clip,
		eikonal_weight=eikonal_weight,
	)

	# Load pretrained model if specified ('random' or None: keep the fresh initialization)
	if load_model_path is not None and load_model_path != 'random':
		print(f'Loading pretrained model from {load_model_path}')
		with open(load_model_path, 'rb') as f:
			saved_data = pickle.load(f)

		# Verify architecture matches
		assert saved_data['embed_dim'] == embed_dim, 'embed_dim mismatch'
		assert saved_data['n_freq'] == n_freq, 'n_freq mismatch'
		assert saved_data['n_heads'] == n_heads, 'n_heads mismatch'
		assert saved_data['n_layers'] == n_layers, 'n_layers mismatch'
		assert saved_data['mlp_dim'] == mlp_dim, 'mlp_dim mismatch'

		# Load parameters into model
		nnx.update(model, saved_data['params'])
		print(f'Successfully loaded pretrained weights from {load_model_path}')

	# Load mesh training data from multiple files.
	all_training_data = []
	t0 = time.time()
	k_neighbors = None
	for training_data_filename in training_data_filenames:
		if not os.path.exists(training_data_filename):
			raise ValueError(f"The path '{training_data_filename}' does not exist")

		print(f"Loading training data from '{training_data_filename}'...")

		data = load_mesh_training_data(training_data_filename)
		if min_points:
			keep = data[3].sum(axis=1) >= min_points
			if not keep.all():
				print(f'  Dropping {int((~keep).sum())} shapes with fewer than {min_points} points')
				data = tuple(x[keep] for x in data)

		(all_positions, all_normals, all_neighbors, all_masks, all_mesh_vertices, all_mesh_faces, all_face_masks) = data

		k_neighbors = all_neighbors.shape[2]
		n_shapes = len(all_positions)
		max_points = all_positions.shape[1]
		print(f'  Loaded {n_shapes} shapes with max {max_points} points each')
		print(f'  k_neighbors: {k_neighbors}')

		all_training_data.append(data)

	t1 = time.time()
	print(f'Load data: {t1 - t0:.6f} s')

	# Create combined list of (dataset_idx, shape_idx) for all shapes (across possibly multiple datasets)
	all_shape_indices = []
	for dataset_idx, data in enumerate(all_training_data):
		n_shapes = len(data[0])  # data[0] is all_positions
		for shape_idx in range(n_shapes):
			all_shape_indices.append((dataset_idx, shape_idx))

	total_shapes = len(all_shape_indices)
	if resample and not multishape:
		raise ValueError('resample needs multi-shape patch batches (multishape > 0)')
	if (noise or quantize) and not multishape:
		raise ValueError('noise and quantize need multi-shape patch batches (multishape > 0)')
	if mix and not multishape:
		raise ValueError('mix needs multi-shape patch batches (multishape > 0)')
	print(f'Total shapes across all datasets: {total_shapes}')

	# Load validation data
	val_data_filename = validation_file
	val_positions = None
	val_normals = None
	val_neighbors = None
	val_masks = None
	val_mesh_vertices = None
	val_mesh_faces = None
	val_face_masks = None

	if os.path.exists(val_data_filename):
		try:
			(
				val_positions,
				val_normals,
				val_neighbors,
				val_masks,
				val_mesh_vertices,
				val_mesh_faces,
				val_face_masks,
			) = load_mesh_training_data(val_data_filename)
			print(f'Loaded validation data: {len(val_positions)} shapes')
		except Exception as e:
			print(f'Warning: Could not load validation data: {e}')
			print('Continuing without validation...')
	else:
		print(f'Validation file not found: {val_data_filename}')

	def save_model(curr_model, epoch, save_filepath: str = None):
		# Save weights as pkl file
		nnx.eval_mode(curr_model)
		params = nnx.state(curr_model, nnx.Param)
		saved_data = {
			'params': params,
			'embed_dim': embed_dim,
			'n_freq': n_freq,
			'n_heads': n_heads,
			'n_layers': n_layers,
			'mlp_dim': mlp_dim,
			'k_neighbors': k_neighbors,
			'dropout_rate': dropout_rate,
		}

		model_name = f'FundamentalFormPredictor_{epoch}' if save_filepath is None else save_filepath
		os.makedirs(models_dir, exist_ok=True)
		filepath = os.path.join(models_dir, model_name + '.pkl')
		with open(filepath, 'wb') as f:
			pickle.dump(saved_data, f)

		print(f'Model saved to {filepath}')

	def run_validation(epoch_key):
		"""Validation loss (distance, eikonality) of the current model; queries from fold_in(epoch_key, 10000 + i)."""
		nnx.eval_mode(model)
		n_val_shapes = len(val_positions)
		val_shape_losses = []
		val_shape_eikonality_losses = []

		for val_shape_idx in range(n_val_shapes):
			# Generate validation key (deterministic per epoch)
			val_key = jax.random.fold_in(epoch_key, 10000 + val_shape_idx)

			val_pos = val_positions[val_shape_idx]
			val_norm = val_normals[val_shape_idx]
			val_neigh = val_neighbors[val_shape_idx]
			val_mask = val_masks[val_shape_idx]
			val_verts = val_mesh_vertices[val_shape_idx]
			val_faces = val_mesh_faces[val_shape_idx]
			val_fmask = val_face_masks[val_shape_idx]

			# Generate validation queries on-the-fly
			val_queries, val_distances = generate_shape_features(
				val_pos,
				val_neigh,
				val_verts,
				val_faces,
				val_fmask,
				val_key,
				n_queries=n_queries_per_neighborhood,
			)

			# Compute weights for this validation shape
			val_weights = model(val_pos[val_neigh], val_norm[val_neigh])

			# Compute loss per batch
			n_val_batches = (val_pos.shape[0] + batch_size - 1) // batch_size
			batch_val_losses = []
			batch_val_eikonality_losses = []

			for batch_idx in range(n_val_batches):
				start_idx = batch_idx * batch_size
				batch_mask = jax.lax.dynamic_slice_in_dim(val_mask, start_idx, batch_size)
				batch_queries = jax.lax.dynamic_slice_in_dim(val_queries, start_idx, batch_size)
				batch_distances = jax.lax.dynamic_slice_in_dim(val_distances, start_idx, batch_size)

				# Only compute if batch has valid points
				if jnp.sum(batch_mask) > 0:
					val_loss, val_eikonality_loss = validation_loss(
						val_mask,
						batch_mask,
						val_pos,
						val_norm,
						val_weights,
						batch_queries,
						batch_distances,
						knn_blend=knn_blend,
					)
					batch_val_losses.append(val_loss)
					batch_val_eikonality_losses.append(val_eikonality_loss)

			# Average over batches for this shape
			if len(batch_val_losses) > 0:
				val_shape_losses.append(np.mean(batch_val_losses))
				val_shape_eikonality_losses.append(np.mean(batch_val_eikonality_losses))

		# Average over all validation shapes
		val_loss = np.mean(val_shape_losses) if len(val_shape_losses) > 0 else 0.0
		val_eikonality_loss = (
			np.mean(val_shape_eikonality_losses) if len(val_shape_eikonality_losses) > 0 else 0.0
		)
		nnx.train_mode(model)
		return val_loss, val_eikonality_loss

	def train_on_patches():
		"""The multi-shape patch loop (multishape > 0); batches are prepared on the CPU one step ahead."""
		if not max_steps:
			raise ValueError('multi-shape patch batches need the number of steps of the phase (max_steps)')
		patch_step = make_patch_step(knn_blend, eikonal_weight)
		rng = np.random.default_rng(seed + 1000)
		stream = []

		mix_fraction, scenes = 0.0, None
		if mix:
			import analytic_scene
			import analytic_sdf

			kind, fraction, path = mix.split(':', 2)
			if kind != 'analytic':
				raise ValueError(f'mix {mix!r}: expected analytic:FRACTION:PATH')
			mix_fraction = float(fraction)
			scenes = dict(np.load(path))
			print(f'Mixing analytic scenes with probability {mix_fraction} ({len(scenes["sigma"])} scenes from {path})')

		def prepare_analytic():
			i = int(rng.integers(len(scenes['sigma'])))
			E, types, polys, pcount = scenes['prims'][i], scenes['types'][i], scenes['polys'][i], scenes['pcount'][i]
			if resample:
				n = int(rng.choice(augment.resample_counts(resample)))
				P, N, _, _, _, _ = analytic_scene.sample_scene(analytic_scene.decode(E, types, polys, pcount), n, rng)
			else:
				P, N = scenes['P'][i], scenes['N'][i]
			R = augment.random_rotation(rng) if rotate else np.eye(3)
			P, N = P @ R.T, N @ R.T
			patch = augment.make_patch(P, N, None, np.ones(len(P), bool), augment.cKDTree(P), k_neighbors, patch_centres, union_size, rng)
			return patch + ('analytic', (E, types, polys, pcount, R))

		def next_shapes():
			nonlocal stream
			while len(stream) < multishape:
				stream += list(np.random.permutation(total_shapes))
			out, stream = stream[:multishape], stream[multishape:]
			return out

		def prepare(global_indices):
			patches = []
			for g in global_indices:
				if mix_fraction > 0 and rng.uniform() < mix_fraction:
					patches.append(prepare_analytic())
					continue
				dataset_idx, shape_idx = all_shape_indices[g]
				arrays = tuple(x[shape_idx] for x in all_training_data[dataset_idx])
				shape_resample = resample if (resample and resample_match in training_data_filenames[dataset_idx]) else ''
				P, N, NB, M, V, F, FM, tree = augment.visit(
					arrays, k_neighbors, rng, shape_resample, rotate, noise, quantize
				)
				patch = augment.make_patch(P, N, NB, M, tree, k_neighbors, patch_centres, union_size, rng)
				patches.append(patch + (V, F, FM))
			return patches

		pool = concurrent.futures.ThreadPoolExecutor(1)
		future = pool.submit(prepare, next_shapes())
		step, log_buffer, union_sizes, t_log = 0, [], [], time.time()
		n_analytic = 0
		while step < max_steps:
			patches = future.result()
			future = pool.submit(prepare, next_shapes())
			step_key = jax.random.fold_in(key, step)
			queries, distances = [], []
			for j, pt in enumerate(patches):
				if isinstance(pt[7], str):  # analytic scene: exact-SDF queries
					E, types, polys, pcount, R = pt[8]
					q, d, _ = analytic_sdf.analytic_queries(
						jnp.asarray(pt[5]),
						jnp.asarray(E),
						jnp.asarray(types),
						jnp.asarray(polys),
						jnp.asarray(pcount),
						jnp.asarray(R, jnp.float32),
						jax.random.fold_in(step_key, j),
						bbox_extension=bbox_extension,
					)
					queries.append(q)
					distances.append(d)
					union_sizes.append(pt[6])
					n_analytic += 1
					continue
				q, d = generate_neighborhood_features(
					jnp.asarray(pt[5]),
					jnp.asarray(pt[7], jnp.float32),
					jnp.asarray(pt[8]),
					jnp.asarray(pt[9]),
					jax.random.fold_in(step_key, j),
					n_queries=n_queries_per_neighborhood,
					narrow_band_offset=narrow_band_offset,
					bbox_extension=bbox_extension,
				)
				queries.append(q)
				distances.append(d)
				union_sizes.append(pt[6])
			stack = lambda i: jnp.asarray(np.stack([pt[i] for pt in patches]))  # noqa: E731
			loss, loss_dict = patch_step(
				model, optimizer, stack(0), stack(1), stack(2), stack(3), stack(4), jnp.stack(queries), jnp.stack(distances)
			)
			step += 1
			log_buffer.append((loss, loss_dict['distance_loss'], loss_dict['eikonality_loss']))
			if (log_every and step % log_every == 0) or step == max_steps:
				v = np.array([[float(x) for x in r] for r in log_buffer])
				now = time.time()
				print(
					f'[step] {step} loss {v[:, 0].mean():.8f} distance {v[:, 1].mean():.8f} eikonality {v[:, 2].mean():.8f} '
					f'{(now - t_log) / len(v):.3f} s/step union mean {np.mean(union_sizes):.0f} max {np.max(union_sizes)}'
					+ (f' analytic patches {n_analytic}' if mix else ''),
					flush=True,
				)
				if not np.all(np.isfinite(v)):
					raise FloatingPointError(f'non-finite loss at step {step}')
				all_train_losses.append(v[:, 0].mean())
				all_distance_losses.append(v[:, 1].mean())
				all_eikonality_losses.append(v[:, 2].mean())
				log_buffer, union_sizes, t_log = [], [], now
			if save_every and step % save_every == 0 and step < max_steps:
				save_model(model, f'step{step}')
				if val_positions is not None:
					val_loss, val_eikonality_loss = run_validation(jax.random.fold_in(key, 0))
					print(f'	{"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
						f'{"eikonality:":<{LABEL_WIDTH}}{val_eikonality_loss:<{VALUE_WIDTH}.6f}', flush=True)
					nnx.train_mode(model)
		pool.shutdown()
		if val_positions is not None:
			val_loss, val_eikonality_loss = run_validation(jax.random.fold_in(key, 0))
			validation_losses.append(val_loss)
			validation_epochs.append(len(all_train_losses) - 1)
			print(f'	{"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
				f'{"eikonality:":<{LABEL_WIDTH}}{val_eikonality_loss:<{VALUE_WIDTH}.6f}', flush=True)

	# Training loop
	nnx.train_mode(model)

	t0 = time.time()
	all_train_losses = []
	all_distance_losses = []
	all_eikonality_losses = []

	validation_losses = []
	validation_epochs = []

	print('\nBeginning training...')
	print(f'  Batch size: {batch_size}')

	num_training_datasets = len(all_training_data)
	np.random.seed(seed)

	rotation_rng = np.random.default_rng(seed + 2000)  # --rotate in the authors' loop
	step = 0  # optimizer steps in this phase
	log_buffer = []
	t_log = time.time()
	stop = False

	if multishape:
		train_on_patches()

	# The authors' loop: every step trains on batch_size query rows of one shape (not used with multishape)
	for epoch in range(0 if multishape else num_epochs):
		epoch_key = jax.random.fold_in(key, epoch)  # deterministic key per epoch

		epoch_losses = []
		epoch_distance_losses = []
		epoch_eikonality_losses = []
		total_num_batches = 0

		# Shuffle all shapes across all datasets together
		shuffled_indices = np.random.permutation(total_shapes)

		for i, global_idx in enumerate(shuffled_indices):
			dataset_idx, shape_idx = all_shape_indices[global_idx]

			# Get data arrays for this dataset
			(
				all_positions,
				all_normals,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			) = all_training_data[dataset_idx]

			max_points = all_positions.shape[1]
			n_batches = (max_points + batch_size - 1) // batch_size

			shape_key = jax.random.fold_in(epoch_key, global_idx)

			# Convert single shape to JAX arrays (GPU)
			positions = jnp.array(all_positions[shape_idx])
			normals = jnp.array(all_normals[shape_idx])
			neighbors = jnp.array(all_neighbors[shape_idx])
			mask = jnp.array(all_masks[shape_idx])
			mesh_vertices = jnp.array(all_mesh_vertices[shape_idx])
			if rotate:
				R = jnp.array(augment.random_rotation(rotation_rng))
				positions, normals, mesh_vertices = positions @ R.T, normals @ R.T, mesh_vertices @ R.T
			mesh_faces = jnp.array(all_mesh_faces[shape_idx])
			face_mask = jnp.array(all_face_masks[shape_idx])

			# Generate query points and ground-truth distances
			shape_queries, shape_distances = generate_shape_features(
				positions,
				neighbors,
				mesh_vertices,
				mesh_faces,
				face_mask,
				shape_key,
				n_queries=n_queries_per_neighborhood,
				narrow_band_offset=narrow_band_offset,
				bbox_extension=bbox_extension,
			)

			batch_losses_list = []
			batch_distance_losses_list = []
			batch_eikonality_losses_list = []

			# Loop over batches for this shape
			for batch_idx in range(n_batches):
				batch_key = jax.random.fold_in(shape_key, batch_idx)
				start_idx = batch_idx * batch_size

				batch_mask = jax.lax.dynamic_slice_in_dim(mask, start_idx, batch_size)
				batch_neighbors = jax.lax.dynamic_slice_in_dim(neighbors, start_idx, batch_size)

				batch_queries = jax.lax.dynamic_slice_in_dim(shape_queries, start_idx, batch_size)
				batch_distances = jax.lax.dynamic_slice_in_dim(shape_distances, start_idx, batch_size)

				# A batch of padded rows only has no loss; an optimizer step on it would still move the weights
				# (Adam momentum with a zero gradient). Skip it.
				if not np.any(all_masks[shape_idx, start_idx : start_idx + batch_size]):
					continue

				# Train step
				loss, loss_dict = train_step(
					model,
					optimizer,
					mask,
					batch_mask,
					positions,
					normals,
					neighbors,
					batch_queries,
					batch_distances,
				)

				batch_losses_list.append(loss)
				batch_distance_losses_list.append(loss_dict['distance_loss'])
				batch_eikonality_losses_list.append(loss_dict['eikonality_loss'])

				step += 1
				if log_every:
					log_buffer.append((loss, loss_dict['distance_loss'], loss_dict['eikonality_loss']))
					if step % log_every == 0:
						v = np.array([[float(x) for x in r] for r in log_buffer])
						log_buffer = []
						now = time.time()
						print(
							f'[step] {step} loss {v[:, 0].mean():.8f} distance {v[:, 1].mean():.8f} '
							f'eikonality {v[:, 2].mean():.8f} {(now - t_log) / log_every:.3f} s/step',
							flush=True,
						)
						t_log = now
				if max_steps and step >= max_steps:
					stop = True
					break

			# Accumulate losses for this shape
			epoch_losses.extend(batch_losses_list)
			epoch_distance_losses.extend(batch_distance_losses_list)
			epoch_eikonality_losses.extend(batch_eikonality_losses_list)
			total_num_batches += len(batch_losses_list)

			# Clean up GPU memory for this shape
			del positions, normals, neighbors, mask
			del mesh_vertices, mesh_faces, face_mask

			if i % n_shapes_to_save == 0 and i > 0:
				epoch_train_loss = np.sum(epoch_losses) / total_num_batches
				epoch_distance_loss = np.sum(epoch_distance_losses) / total_num_batches
				epoch_eikonality_loss = np.sum(epoch_eikonality_losses) / total_num_batches

				print(
					f'  Epoch {epoch}, shape {i}/{total_shapes}: '
					f'loss = {epoch_train_loss:.6f}  '
					f'distance = {epoch_distance_loss:.6f}  '
					f'eikonality = {epoch_eikonality_loss:.6f}'
				)

			if stop:
				break

		# Compute epoch averages
		epoch_train_loss = np.sum(epoch_losses) / total_num_batches
		epoch_distance_loss = np.sum(epoch_distance_losses) / total_num_batches
		epoch_eikonality_loss = np.sum(epoch_eikonality_losses) / total_num_batches

		all_train_losses.append(epoch_train_loss)
		all_distance_losses.append(epoch_distance_loss)
		all_eikonality_losses.append(epoch_eikonality_loss)

		if epoch % n_epochs_to_save == 0:
			print(
				f'Epoch {epoch}:\n'
				f'	{"total loss:":<{LABEL_WIDTH}}{epoch_train_loss:<{VALUE_WIDTH}.6f}'
				f'{"distance:":<{LABEL_WIDTH}}{epoch_distance_loss:<{VALUE_WIDTH}.6f}'
				f'{"eikonality:":<{LABEL_WIDTH}}{epoch_eikonality_loss:<{VALUE_WIDTH}.6f}'
			)

			# Save model checkpoint
			save_model(model, epoch)

			# Validation loss
			if val_positions is not None:
				val_loss, val_eikonality_loss = run_validation(epoch_key)

				validation_losses.append(val_loss)
				validation_epochs.append(epoch)

				print(
					f'	{"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
					f'{"eikonality:":<{LABEL_WIDTH}}{val_eikonality_loss:<{VALUE_WIDTH}.6f}'
				)

		if stop:
			break

	t1 = time.time()
	print(f'Total training time: {t1 - t0:.2f} s')

	train_losses = np.array(all_train_losses)
	distance_losses = np.array(all_distance_losses)
	eikonality_losses = np.array(all_eikonality_losses)

	# Save final model
	save_model(model, num_epochs - 1, final_save_filepath)

	# Plot training history

	fig, axes = plt.subplots(2, 2, figsize=(15, 10))

	# Total training loss
	axes[0, 0].plot(train_losses, label='Total Training Loss', color='blue')
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Loss')
	axes[0, 0].set_title('Total Training Loss')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	# Distance loss
	axes[0, 1].plot(distance_losses, label='Distance Error', color='green')
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('Distance Error')
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	# Eikonality loss
	axes[1, 0].plot(eikonality_losses, label='Eikonality Error', color='orange')
	axes[1, 0].set_xlabel('Epoch')
	axes[1, 0].set_ylabel('Loss')
	axes[1, 0].set_title('Eikonality Error')
	axes[1, 0].legend()
	axes[1, 0].grid(True, alpha=0.3)

	# Validation loss
	if len(validation_losses) > 0:
		axes[1, 1].plot(validation_epochs, validation_losses, label='Validation Loss', color='red', marker='o')
		axes[1, 1].set_xlabel('Epoch')
		axes[1, 1].set_ylabel('Loss')
		axes[1, 1].set_title('Validation Loss')
		axes[1, 1].legend()
		axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('PlotTrainingHistory.png', dpi=150)
	print('Training history plot saved to PlotTrainingHistory.png')


def authors_curriculum(data_dir='datasets', models_dir=MODELS_DIR):
	"""
	The training curriculum of the paper (Sec. 4.3; plus the pretraining on a single blob), as a list of phases.
	Each phase is a dict of TrainFundamentalFormPredictor arguments plus its 'name'; a phase starts from the model saved
	by the previous one.
	"""
	dataset_dir = os.path.join(data_dir, 'nice')
	simulated_dir = os.path.join(data_dir, 'simulated')
	nice = lambda pattern: glob.glob(os.path.join(dataset_dir, pattern))  # noqa: E731
	saved = lambda name: os.path.join(models_dir, name + '.pkl')  # noqa: E731

	# Phase 1 interleaves the blob and ABC lists
	phase1_files = [val for tup in zip(nice('blobs_train_mode0*.npz'), nice('ABC17_train_mode0_*.npz')) for val in tup]

	return [
		# Pretraining
		dict(
			name='pretrain',
			training_data_filenames=[os.path.join(data_dir, 'pretraining_blob.npz')],
			num_epochs=64,
			learning_rate=1e-3,
			warmup_steps=0,
			batch_size=32,
			n_epochs_to_save=1,
			n_shapes_to_save=1,
			load_model_path=None,
			final_save_filepath='FundamentalFormPredictor_pretrained',
		),
		# Phase 0: nicely sampled blobs
		dict(
			name='phase0',
			training_data_filenames=sorted(nice('blobs_train_mode0*.npz')),
			num_epochs=24,
			learning_rate=1e-5,
			warmup_steps=1e2,
			batch_size=1024,
			n_epochs_to_save=1,
			n_shapes_to_save=1,
			load_model_path=saved('FundamentalFormPredictor_pretrained'),
			final_save_filepath='FundamentalFormPredictor_phase0',
		),
		# Phase 1: nicely sampled blobs and ABC meshes
		dict(
			name='phase1',
			training_data_filenames=phase1_files,
			num_epochs=8,
			learning_rate=1e-5,
			warmup_steps=1e3,
			batch_size=1024,
			n_epochs_to_save=1,
			n_shapes_to_save=100,
			load_model_path=saved('FundamentalFormPredictor_phase0'),
			final_save_filepath='FundamentalFormPredictor_phase1',
		),
		# Phase 2: nicely sampled blobs and more ABC meshes
		dict(
			name='phase2',
			training_data_filenames=nice('blobs_train_mode0*.npz') + nice('ABC17_train_mode0_*.npz') + nice('ABC2_train_mode0_*.npz'),
			num_epochs=8,
			learning_rate=1e-6,
			warmup_steps=1e3,
			batch_size=1024,
			n_epochs_to_save=1,
			n_shapes_to_save=100,
			load_model_path=saved('FundamentalFormPredictor_phase1'),
			final_save_filepath='FundamentalFormPredictor_phase2',
		),
		# Phase 3: nicely sampled blobs and even more ABC meshes
		dict(
			name='phase3',
			training_data_filenames=nice('blobs_train_mode0*.npz') + nice('ABC*_train_mode0_*.npz'),
			num_epochs=5,
			learning_rate=1e-6,
			warmup_steps=1e3,
			batch_size=1024,
			n_epochs_to_save=1,
			n_shapes_to_save=100,
			load_model_path=saved('FundamentalFormPredictor_phase2'),
			final_save_filepath='FundamentalFormPredictor_phase3',
		),
		# Phase 4: everything from before, plus unevenly sampled data possibly with small gaps from a simulated scanning process
		dict(
			name='phase4',
			training_data_filenames=nice('blobs_train_mode0*.npz')
			+ nice('ABC*_train_mode0_*.npz')
			+ glob.glob(os.path.join(simulated_dir, 'blobs_train_mode*.npz'))
			+ glob.glob(os.path.join(simulated_dir, 'ABC*_train_mode*.npz')),
			num_epochs=2,
			learning_rate=1e-7,
			warmup_steps=1e3,
			batch_size=1024,
			n_epochs_to_save=1,
			n_shapes_to_save=100,
			load_model_path=saved('FundamentalFormPredictor_phase3'),
			final_save_filepath='FundamentalFormPredictor_phase4',
		),
	]


def custom_curriculum(phase_specs, data_dir='datasets', models_dir=MODELS_DIR):
	"""
	Phases given as 'NAME:GLOBS:STEPS' strings: GLOBS is a comma-separated list of file patterns relative to data_dir
	(each sorted, then concatenated), STEPS the number of optimizer steps of the phase. The other settings are those of
	the authors' phase 1 (lr 1e-5, 1000 warmup steps, batch size 1024); the first phase starts from a fresh
	initialization, every later one from the model saved by the previous phase.
	"""
	phases, previous = [], None
	for spec in phase_specs:
		name, patterns, steps = spec.split(':')
		files = []
		for pattern in patterns.split(','):
			matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
			if not matches:
				raise ValueError(f"--phase {spec}: no file matches '{os.path.join(data_dir, pattern)}'")
			files += matches
		phases.append(
			dict(
				name=name,
				training_data_filenames=files,
				num_epochs=1_000_000,
				learning_rate=1e-5,
				warmup_steps=1e3,
				batch_size=1024,
				n_epochs_to_save=1,
				n_shapes_to_save=100,
				load_model_path=previous,
				final_save_filepath=f'FundamentalFormPredictor_{name}',
				max_steps=int(steps),
			)
		)
		previous = os.path.join(models_dir, f'FundamentalFormPredictor_{name}.pkl')
	return phases


def train(
	data_dir='datasets',
	models_dir=MODELS_DIR,
	only_phases=None,
	phase_specs=None,
	init=None,
	skip_pretraining=False,
	max_steps=0,
	learning_rate=None,
	warmup_steps=None,
	gate=False,
	**options,
):
	"""
	Run the curriculum: the authors' (default) or the phases phase_specs (see custom_curriculum).
	only_phases: names of the phases to run (None: all). learning_rate, warmup_steps: override every phase value (None:
	the curriculum value). gate: after every phase, evaluate its model with eval_model.py and print the gate checks
	(log only). init: weights of the first phase that runs ('random' or a
	.pkl path; None: the curriculum's own). skip_pretraining: drop the authors' pretraining; phase 0 then starts from
	init (default: a fresh initialization). max_steps: cap on the steps of every phase (0: none).
	options: further keyword arguments of TrainFundamentalFormPredictor, the same for every phase.
	"""
	if phase_specs:
		phases = custom_curriculum(phase_specs, data_dir, models_dir)
	else:
		phases = authors_curriculum(data_dir, models_dir)
		if skip_pretraining:
			phases = [ph for ph in phases if ph['name'] != 'pretrain']
			phases[0]['load_model_path'] = None
	if only_phases:
		phases = [ph for ph in phases if ph['name'] in only_phases]
	for i, phase in enumerate(phases):
		name = phase.pop('name')
		if i == 0 and init is not None:
			phase['load_model_path'] = init
		phase_steps = phase.pop('max_steps', 0)
		if learning_rate is not None:
			phase['learning_rate'] = learning_rate
		if warmup_steps is not None:
			phase['warmup_steps'] = warmup_steps
		steps = min(x for x in (phase_steps, max_steps) if x) if (phase_steps or max_steps) else 0
		print(f'\n===== {name} =====', flush=True)
		TrainFundamentalFormPredictor(**phase, models_dir=models_dir, max_steps=steps, **options)
		if gate:
			import eval_model

			path = os.path.join(models_dir, phase['final_save_filepath'] + '.pkl')
			res = eval_model.evaluate(
				path, label=name, out_dir=os.path.join(models_dir, 'eval'), validation_file=options.get('validation_file')
			)
			print(f'\n===== gate {name} ({path}) =====\n{eval_model.HEAD}\n{eval_model.row(res)}', flush=True)
			lines, passed = eval_model.gate_report(res)
			print('\n'.join(lines) + f'\n[gate] {name}: {"all checks pass" if passed else "some checks MISS"}', flush=True)


def parse_args(argv=None):
	ap = argparse.ArgumentParser(
		description='Train the Points-as-Tori fundamental-form predictor. With no options this runs the curriculum of '
		'the paper (pretraining, phases 0-4) from ./datasets, saving to ./models.'
	)
	ap.add_argument('--data-dir', default='datasets', help='training data directory (nice/, simulated/, ...)')
	ap.add_argument('--models-dir', default=MODELS_DIR, help='where models are saved and loaded between phases')
	ap.add_argument(
		'--validation-file', default=None, help='validation data (default: DATA_DIR/ABC17_validation_3.npz)'
	)
	ap.add_argument('--seed', type=int, default=42)
	ap.add_argument(
		'--phase',
		action='append',
		default=[],
		metavar='NAME:GLOBS:STEPS',
		help='run this phase instead of the authors\' curriculum (repeatable, in order); GLOBS: comma-separated file '
		'patterns relative to --data-dir; each phase starts from the previous one',
	)
	ap.add_argument(
		'--init', default=None, help="initial weights of the first phase: 'random' or a .pkl (default: the curriculum's)"
	)
	ap.add_argument(
		'--skip-pretraining',
		action='store_true',
		help='drop the pretraining phase of the authors\' curriculum; phase 0 starts from --init (default: random)',
	)
	ap.add_argument(
		'--only-phases', default='', help='comma-separated subset of the phases to run: pretrain,phase0,...,phase4'
	)
	ap.add_argument('--max-steps', type=int, default=0, help='stop each phase after this many steps (0: all epochs)')
	ap.add_argument(
		'--gate',
		action='store_true',
		help='after every phase, evaluate its model (eval_model.py, about 2 min) and print the gate checks; log only',
	)
	ap.add_argument('--log-every', type=int, default=0, help='print the mean loss every N steps (0: off)')
	ap.add_argument('--lr', type=float, default=None, help="peak learning rate of every phase (default: the curriculum's)")
	ap.add_argument('--warmup', type=int, default=None, help="linear warmup steps of every phase (default: the curriculum's)")
	ap.add_argument(
		'--schedule',
		choices=['authors', 'cosine'],
		default='authors',
		help="authors: cosine decay over 1e8 steps (constant in practice); cosine: decay to 1 %% of the lr over each phase's steps",
	)
	ap.add_argument(
		'--multishape',
		type=int,
		default=0,
		metavar='S',
		help='train on multi-shape patch batches: one patch of --patch-c clustered centres from each of S shapes per step '
		'(needs a step count: --phase ...:STEPS or --max-steps; 0: the authors\' one-shape batches)',
	)
	ap.add_argument('--patch-c', type=int, default=32, help='query centres per patch (--multishape)')
	ap.add_argument('--upad', type=int, default=256, help='patch union size, padded (--multishape)')
	ap.add_argument(
		'--save-every', type=int, default=10000, help='checkpoint and validation every N steps (--multishape; 0: at the end)'
	)
	ap.add_argument(
		'--resample',
		default='',
		metavar='MIN:MAX',
		help='with --multishape: fresh points on the ground-truth mesh at every visit, count uniform among the powers of two '
		'in [MIN, MAX], k-NN recomputed (default: the stored points)',
	)
	ap.add_argument(
		'--resample-match', default='', help='resample only shapes whose data file path contains this string (default: all)'
	)
	ap.add_argument(
		'--rotate', action='store_true', help='random rotation of every shape visit (points, normals, mesh)'
	)
	ap.add_argument(
		'--noise',
		type=float,
		default=0.0,
		metavar='S',
		help='with --multishape: Gaussian noise of S sigma on the input points (sigma: median k-NN distance; 0: off)',
	)
	ap.add_argument(
		'--quantize',
		type=float,
		default=0.0,
		metavar='Q',
		help='with --multishape: quantize the input points along a random direction per visit to steps of Q sigma (0: off)',
	)
	ap.add_argument(
		'--mix',
		default='',
		metavar='analytic:FRAC:PATH',
		help='with --multishape: each patch is, with probability FRAC, from an analytic scene of PATH (gen_analytic.py) '
		'with exact-SDF queries (default: off)',
	)
	ap.add_argument('--eik-w', type=float, default=1.0, help='weight of the eikonality term of the loss')
	ap.add_argument('--clip', type=float, default=0.0, help='clip the gradient to this global norm (0: off)')
	ap.add_argument('--weight-decay', type=float, default=0.0, help='AdamW weight decay')
	ap.add_argument(
		'--min-points', type=int, default=0, help='drop training shapes with fewer valid points (0: keep all)'
	)
	ap.add_argument(
		'--knn-blend',
		type=int,
		default=0,
		help='blend the tori over the K nearest points of each query, as the accelerated inference does (0: all points)',
	)
	return ap.parse_args(argv)


def main(argv=None):
	a = parse_args(argv)
	train(
		data_dir=a.data_dir,
		models_dir=a.models_dir,
		only_phases=[x for x in a.only_phases.split(',') if x] or None,
		phase_specs=a.phase,
		init=a.init,
		skip_pretraining=a.skip_pretraining,
		gate=a.gate,
		seed=a.seed,
		validation_file=a.validation_file or os.path.join(a.data_dir, 'ABC17_validation_3.npz'),
		max_steps=a.max_steps,
		log_every=a.log_every,
		knn_blend=a.knn_blend,
		min_points=a.min_points,
		learning_rate=a.lr,
		warmup_steps=a.warmup,
		schedule=a.schedule,
		clip=a.clip,
		weight_decay=a.weight_decay,
		eikonal_weight=a.eik_w,
		multishape=a.multishape,
		patch_centres=a.patch_c,
		union_size=a.upad,
		save_every=a.save_every,
		resample=a.resample,
		resample_match=a.resample_match,
		rotate=a.rotate,
		noise=a.noise,
		quantize=a.quantize,
		mix=a.mix,
	)


if __name__ == '__main__':
	main()
