from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
import numpy as np

class TopicProportions():
	def __init__(self):
		self.tfidf = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)

	def wrapper(self, documents):
		"""Wrapper around calculating TFIDF and finding the tokens to remove.
		Args:
			documents (list, str): List of preprocessed meetup tags.
		Returns:
			df_remove_tokens"""
		matrix_tfidf, f = self.docs2tdidf(documents)
		df_tfidf = self.matrix2df(matrix_tfidf, f)
		tokens = self.garbage_tokens(documents, df_tfidf)
		return tokens


	def docs2tdidf(self, documents):
		"""Calculate the TFIDF values of the tokens in the corpus.

		Args:
			documents (list, str): Preprocessed tokens.
		Return:
			Sparse matrix with TFIDF values.
		"""
		tfidf = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
		result_tfidf = tfidf.fit(documents)
		# Word mapping
		value_index = result_tfidf.vocabulary_
		f = {v:k for k, v in value_index.items()}
		# Transform the original documents
		transformed = tfidf.transform(documents)
		return transformed, f

	def matrix2df(self, matrix_tfidf, f):
		"""Return a COOrdinate representation of this matrix and store it in a dataframe"""

		# matrix_tfidf, f = self.docs2tdidf(documents)

		coo = matrix_tfidf.tocoo(copy=False)
		x = pd.DataFrame({'index': coo.row, 'col': coo.col, 'data': coo.data}
						 )[['index', 'col', 'data']].sort_values(['index', 'col']
						 ).reset_index(drop=True)

		df = x.pivot(index='index', columns='col', values='data')
		df.rename(index=str, inplace=True, columns=f)
		df.reset_index(inplace=True)
		df.fillna(0, inplace=True)

		return df

	def garbage_tokens(self, documents, df_tfidf):#, df_groups):
		"""Find the tokens of each document to be deleted.
		"""
		garbage  = []
		for i, tag_tokens in enumerate(list(documents)):
			for tag_token in tag_tokens:
				try:
					if df_tfidf.loc[i, tag_token] > 0 and df_tfidf.loc[i, tag_token] < 0.1:
						garbage.append(tag_token)
				except KeyError as k:
					continue

		return [w for w in garbage if w != 'big_data']

	def remove_items(self, lsts1, lsts2):
		"""A really dirty way of removing elements from a list, if they are found in another list."""
		elements = []
		for i, elems in enumerate(lsts1):
			x = []
			for elem in elems:
				if elem not in lsts2:
					x.append(elem)
				else:
					continue
			elements.append(x)
		return elements

	def find_token_labels(self, df, clustered_tokens):
		# Find the labels of each group
		group_labels = []
		for tokens in df.reduced_tags:
			lbl = []
			for token in tokens:
				for clustered_token in clustered_tokens:
					if token == clustered_token[0]:
						lbl.append(clustered_token[1])

			group_labels.append(lbl)

		return group_labels

	def topic_proportions(self, df, group_labels):
		# Find reduced topic proportions
		group_topics = {}
		for i, group_name in enumerate(list(df.name)):
			# Proportions of tokens
			proportions = {}
			c = dict(Counter(group_labels[i]))
			for key in c.keys():
				proportions[key] = c[key] / np.sum(list(c.values()))

			group_topics[group_name] = proportions

		df_topics_proportions = pd.DataFrame.from_dict(group_topics).T
		df_topics_proportions.fillna(0, inplace=True)
		df_topics_proportions.reset_index(inplace=True)

		return df_topics_proportions

def process_tags(tags_groups):
	"""Simple text preprocessing that keeps tags in a nice format. If you are seeing this comment, please change it."""
	texts = [[token.lower() for token in re.findall("\'(.*?)\'", tags_group)] for tags_group in tags_groups]
	return [[re.sub('\s', '_', txt) for txt in text] for text in texts]
