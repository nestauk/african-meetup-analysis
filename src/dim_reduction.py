import numpy as np
import scipy
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import mixture

class WrapTSNE():
	def __init__(self, w2v):
		self.w2v = w2v

	def word_vectors(self):
		"""Find the N-dimensional vector of each word.

		Args:
			w2v: Trained word2vec model.

		Return:
			token_vectors (dict): Dictionary where keys are the tokens and values the N-dim vectors.

		"""
		return {w:self.w2v[w] for w in self.w2v.wv.vocab}

	def vectors2sparse_matrix(self, vectors):
		"""Transform a vector to a sparse matrix.

		Args:
			vectors (array): The set of N-dimensional vectors.

		Return:
			A sparse matrix.

		"""
		return scipy.sparse.csr_matrix(vectors, dtype = 'double')

	def unravel_dictionary(self, token_vectors):
		"""Extract the values of a dictionary.

		Args:
			token_vectors (dict): Dictionary where keys are the tokens and values the N-dim vectors.

		Return:
			dict_keys (list, str): The keys of a dictionary.
			dict_values (array, float): The values of a dictionary in a Numpy array.

		"""
		dict_keys = list(token_vectors.keys())
		dict_values = np.array(list(token_vectors.values()))

		return dict_keys, dict_values

	def calculate_cosine_similarity(self, sparse_matrix):
		"""Calculate the cosine similarity of a sparse matrix."""
		return cosine_similarity(sparse_matrix)

	def calculate_cosine_distance(self, similarities):
		"""Find the cosine distance of some entities. Cosine distance = 1 - cosine similarity.

		Args:
			similarities (array): The cosine similarity.

		Return:
			cosine_distance (array): The cosine distance.

		"""
		cos_distance = 1 - similarities
		print(np.clip(cos_distance,0,1,cos_distance))
		return np.clip(cos_distance,0,1,cos_distance)

	def cos_dis(self, vectors):
		"""Wrapper function that finds the cosine distance of given vectors.

		Args:
			vectors (array): The set of N-dimensional vectors.

		Return:
			Cosine distance.

		"""
		return self.calculate_cosine_distance(self.calculate_cosine_similarity(self.vectors2sparse_matrix(vectors)))

	def reduce_dimensions(self, n_iter=1500, perplexity=50):
		"""Wrapper function that transforms word vectors to 2D."""
		dict_keys, dict_values = self.unravel_dictionary(self.word_vectors())
		cos_dist = self.cos_dis(dict_values)
		tsne = TSNE(n_components=2, random_state=0, verbose=1, n_iter=n_iter, perplexity=perplexity, metric='precomputed')
		return tsne.fit_transform(cos_dist)


class GME():
	def __init__(self, data):
		self.data = data

	def fit_eval(self, max_components):
		"""Evaluate GMM through BIC and keep the best model.

		Args:
			data (array): 2D space produced by t-SNE.

		Return:
			best_gmm: GMM with the lowest BIC.
			bic (array): The BIC values for the various GMMs that were produced.

		"""
		lowest_bic = np.infty
		bic = []
		n_components_range = range(1, max_components)
		cv_types = ['spherical', 'tied', 'diag', 'full']
		for cv_type in cv_types:
			for n_components in n_components_range:
				# Fit a Gaussian mixture model
				gmm = mixture.GaussianMixture(n_components=n_components,
											  covariance_type=cv_type)
				gmm.fit(self.data)
				bic.append(gmm.bic(self.data))
				if bic[-1] < lowest_bic:
					lowest_bic = bic[-1]
					best_gmm = gmm

		bic = np.array(bic)
		return best_gmm, bic
