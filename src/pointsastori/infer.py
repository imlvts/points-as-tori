from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional, Tuple

from .shape_3d import TorusDistanceField, PointCloud3D, fit_tori_from_forms
from .network import FundamentalFormPredictor

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FundamentalFormPredictor.pkl')
K_NEIGHBORS_ACCELERATION = 32


class PointsAsTori:
	"""
	Object that stores an oriented point cloud and its (pre-)fitted tori.

	Usage:

		pat = PointsAsTori(points, normals, model_path='path/to/model.pkl')
		distances = pat.signed_distance(queries)   # (n_queries, ) array
		gradients = pat.sdf_gradient(queries)	  # (n_queries, 3) array
	"""

	def __init__(
		self,
		points: np.ndarray,
		normals: np.ndarray,
		tori: dict = None,
		tori_filepath: str = None,
		model_path: str = None,
		frame: str = None,
	) -> None:
		"""
		Args:
			points: (|P|, 3) array of point positions
			normals: (|P|, 3) array of point normals
			tori: dictionary of tori, has format
				 {centers: (|P|, 3) NumPy array, axes: (|P|, 2) NumPy array, major_radii: (|P|,) NumPy array, minor_radii: (|P|,) NumPy array}
			tori_filepath: filepath to precomputed tori
			model_path: path to a .pkl model file representing a trained neural network.
				Defaults to the included pre-trained model.
			frame: None (the model's own) or 'invariant' / 'continuous' / 'legacy': evaluate the network in that tangent
				frame (network.FRAMES). 'invariant' averages three frames attached to each neighbourhood: exactly
				rotation-invariant tori, 3x the network cost. Off by default.
		"""
		self._tdf = TorusDistanceField(points, normals)

		if tori_filepath is not None:
			tori = self._load_tori(tori_filepath)
			self._tdf.set_tori(tori['centers'], tori['axes'], tori['major_radii'], tori['minor_radii'])

		if tori is None:
			if model_path is None:
				model_path = DEFAULT_MODEL_PATH
			if not os.path.isfile(model_path):
				raise FileNotFoundError(f'Model file not found: {model_path}')

			model, k_nb = PointsAsTori._load_model(model_path, frame)

			coeffs, aux = model.precompute_coefficients_in_chunks(points, normals, k_nb, chunk_size=50000, return_aux=True)
			centers, axes, major_radii, minor_radii = fit_tori_from_forms(points, normals, np.array(coeffs))
			if aux is not None:
				minor_radii = self._apply_aux_heads(aux, np.array(minor_radii))

			self._tdf.set_tori(centers, axes, major_radii, minor_radii)

		else:
			self._tdf.set_tori(tori['centers'], tori['axes'], tori['major_radii'], tori['minor_radii'])

	def _apply_aux_heads(self, aux: np.ndarray, minor_radii: np.ndarray) -> np.ndarray:
		"""
		The extra outputs of a model trained with --aux-heads, as in training (network.aux_blend_terms):
		flip logit < 0: the point's normal is taken as flipped, i.e. its torus' SDF is negated (the C++ fields encode the
		sign in the minor radius); confidence alpha = sigmoid(logit) multiplies the point's blend weight (C++
		set_point_weights; bindings without it fall back to marking points with alpha < 0.05 max alpha as outliers).
		Tori saved with save_tori keep the flips but not the confidences.
		"""
		flip = aux[:, 0] < 0
		alpha = 1.0 / (1.0 + np.exp(-aux[:, 1].astype(np.float64)))
		if self._tdf.has_point_weights():
			self._tdf.set_point_weights(alpha)
		else:
			print('Warning: the pat_bindings in use have no set_point_weights; thresholding the confidences instead')
			self._tdf.set_outliers(np.nonzero(alpha < 0.05 * alpha.max())[0].astype(np.int32))
		print(f'aux heads: {int(flip.sum())} flipped normals, confidence min {alpha.min():.3g} median {np.median(alpha):.3g}')
		return np.where(flip, -minor_radii, minor_radii)

	def signed_distance(self, queries: np.ndarray, accelerate: bool = True) -> np.ndarray:
		"""
		Evaluate signed distance at the given query points.

		Args:
			queries: query locations, shape (n_queries, 3)

		Returns:
			(n_queries, ) array of distances
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_distance(queries)

	def sdf_gradient(self, queries: np.ndarray, accelerate: bool = True) -> np.ndarray:
		"""
		Evaluate the gradient of signed distance at the given query points.

		Args:
			queries: query locations, shape (n_queries, 3)

		Returns:
			(n_queries, 3) array
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_gradient(queries)

	def signed_distance_and_gradient(
		self, queries: np.ndarray, accelerate: bool = True
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Return (distances, gradients) at the given query points.
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		distances, gradients, _ = self._tdf.evaluate_distance_gradient_laplacian(queries)
		return distances, gradients

	@staticmethod
	def _load_model(model_path: str, frame: str = None):
		model, k_neighbors = FundamentalFormPredictor.load_saved_model(model_path, frame=frame)
		return model, k_neighbors

	def save_tori(self, filepath: str = 'tori/tori.pkl') -> None:
		"""
		Save tori as binary file.

		Args:
			tori: dictionary of tori, has format
				  {centers: (|P|, 3) NumPy array, axes: (|P|, 2) NumPy array, major_radii: (|P|,) NumPy array, minor_radii: (|P|,) NumPy array}
			filepath: output filepath
		"""
		dirpath = os.path.dirname(filepath)
		if dirpath:
			os.makedirs(dirpath, exist_ok=True)
		tori = self._tdf.get_tori()
		with open(filepath, 'wb') as f:
			pickle.dump(tori, f)

	def _load_tori(self, filepath: str = 'tori/tori.pkl') -> dict:
		"""
		Load tori from a binary file saved by save_tori().

		Args:
			filepath: path to the tori file

		Returns:
			dictionary with keys 'centers', 'axes', 'major_radii', 'minor_radii'
		"""
		with open(filepath, 'rb') as f:
			tori = pickle.load(f)  # tuple
			return {'centers': tori[0], 'axes': tori[1], 'major_radii': tori[2], 'minor_radii': tori[3]}


def read_point_cloud(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load an oriented point cloud from a PLY or OBJ file.

	Args:
		filepath: location of point cloud file

	Returns:
		points: (N, 3) array
		normals: (N, 3) array
	"""
	pc = PointCloud3D.read(filepath)
	return pc.points, pc.normals
