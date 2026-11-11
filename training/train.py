import argparse

import matplotlib.pyplot as plt

from pointsastori.shape_3d import *
from pointsastori.network import *
from mesh_data import *
from mesh_sampling_jax import *


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


@jit
def SDF_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
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
	"""

	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii
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

	loss = distance_loss + eikonality_loss

	# Return individual components for logging
	loss_dict = {
		'distance_loss': distance_loss,
		'eikonality_loss': eikonality_loss,
	}

	return loss, loss_dict


@jit
def validation_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
):
	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii
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
):
	"""
	Set up network (model) with the given parameters.

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
			# optax.clip_by_global_norm(1.0),
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

			loss, loss_dict = SDF_loss(mask, batch_mask, positions, normals, coefficients, queries, true_distances)

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
):
	"""
	Args:
		load_model_path: Optional path to a saved model file (.pkl) to use as warm start
		seed: seed of the network initialization, the shape order and the query sampling
		models_dir: directory of the saved models
		validation_file: validation data (skipped if the file does not exist)
		max_steps: stop after this many optimizer steps (0: run all epochs)
		log_every: print the mean loss of the last log_every steps (0: off)
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
	weight_decay = 0.0
	total_steps = 1e8

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
	)

	# Load pretrained model if specified
	if load_model_path is not None:
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

	step = 0  # optimizer steps in this phase
	log_buffer = []
	t_log = time.time()
	stop = False

	for epoch in range(num_epochs):
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


def train(data_dir='datasets', models_dir=MODELS_DIR, only_phases=None, **options):
	"""
	Run the curriculum. only_phases: names of the phases to run (None: all); options: further keyword arguments of
	TrainFundamentalFormPredictor, the same for every phase.
	"""
	for phase in authors_curriculum(data_dir, models_dir):
		name = phase.pop('name')
		if only_phases and name not in only_phases:
			continue
		print(f'\n===== {name} =====', flush=True)
		TrainFundamentalFormPredictor(**phase, models_dir=models_dir, **options)


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
		'--only-phases', default='', help='comma-separated subset of the phases to run: pretrain,phase0,...,phase4'
	)
	ap.add_argument('--max-steps', type=int, default=0, help='stop each phase after this many steps (0: all epochs)')
	ap.add_argument('--log-every', type=int, default=0, help='print the mean loss every N steps (0: off)')
	return ap.parse_args(argv)


def main(argv=None):
	a = parse_args(argv)
	train(
		data_dir=a.data_dir,
		models_dir=a.models_dir,
		only_phases=[x for x in a.only_phases.split(',') if x] or None,
		seed=a.seed,
		validation_file=a.validation_file or os.path.join(a.data_dir, 'ABC17_validation_3.npz'),
		max_steps=a.max_steps,
		log_every=a.log_every,
	)


if __name__ == '__main__':
	main()
