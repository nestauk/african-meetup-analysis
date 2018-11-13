import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

class Tfidf():
    """Calculates the TF-IDF weighting of a corpus."""
    def __init__(self, docs):
        """Args:
            docs (list, str): Nested lists with preprocessed tokens. Example: [['foo', 'bar'], ['alpha', 'beta']]

        """
        self.documents = docs

    def tfidf_transformer(self):
        """Calculates the TF-IDF. The TfidfVectorizer() passes the tokens as they are, without preprocessing.
        
        Args:
            self.documents (list, str): Nested lists with preprocessed tokens.
            
        Returns:
            tfidf: Trained instance of the sklearn TfidfVectorizer.
            X (csr_matrix): Sparse matrix with the IDF weighting of the tokens and shape (documents, tokens).
            feature_names (list, str): Unique tokens used in TfidfVectorizer. 
        
        """
        tfidf = TfidfVectorizer(tokenizer=lambda x:x, lowercase=False)
        X = tfidf.fit_transform(self.documents)
        feature_names = tfidf.get_feature_names()
        return tfidf, X, feature_names

class WordQueries(Tfidf):
    """Expand the query term list with words similar to the original input. The word similarity is found based on pretrained vectors and TFIDF weighting is used to filter out rare and very frequent terms from the list."""
    def __init__(self, word2vec, docs):
        """Args:
            w2v (word2vec): Pretrained word vectors.
            tfidf: Trained instance of the sklearn TfidfVectorizer.
            
        """
        super().__init__(docs)
        self.w2v = word2vec
        self.tfidf, _, _ = self.tfidf_transformer()

    def similar_tokens(self, token, sim_tokens=25):
        """Find the most similar tokens using a pretrained word2vec model. The top 25 words are considered.

        Args:
            token (str): Query token.

        Returns:
            tokens (set, str): Top 25 most similar to the input token. The input term is also added to the set.

        """
        # Add the query word directly into the list because it will not be in the similar ones.
        tokens = [token]
        tokens.extend([tup[0] for tup in self.w2v.most_similar([token], topn=sim_tokens)])
        return set(tokens)

    def word2ids(self, tokens):
        """Find the TF-IDF IDs of tokens. 

        Args:
            tokens (set, str): Unique, preprocessed tokens.

        Returns:
            token_ids(list, int): Token IDs of the TF-IDF dictionary.

        """
        token_ids = []
        for token in set(tokens):
            try:
                token_ids.append(self.tfidf.vocabulary_[token])
            except Exception as e:
                continue
                
        return token_ids

    def high_idf_ids(self, token_ids, bottom_lim, upper_lim):
        """Filter out tokens with extreme TF-IDF weights (both rare and frequent tokens).

        Args:
            token_ids: (list, int): Token IDs of the TF-IDF dictionary.

        Returns:
            ids: (list, int): Token IDs of the TF-IDF dictionary.

        """
        idfs = {token_id:self.tfidf.idf_[token_id] for token_id in token_ids}
        vals = [val for val in idfs.values()]
        # Keeping only IDs of tokens that have score between the [35,95] of the IDF distribution.
        ids = [k for k, v in idfs.items() 
               if v >= np.percentile(vals, [bottom_lim])[0]
               and v < np.percentile(vals, [upper_lim])[0]]
        return ids

    def tfidf_id2word(self, id_):
        """Find a token using its TF-IDF ID.

        Args:
            id_ (int): TF-IDF id of a token.

        Returns:
            Token (str) that corresponds to the TF-IDF id.

        """
        return list(self.tfidf.vocabulary_.keys())[list(self.tfidf.vocabulary_.values()).index(id_)]

    def query_word(self, token, sim_tokens=25, bottom_lim=35, upper_lim=95):
        """Wrapper function around the WordQueries class.

        Args:
            token (str): Token to query the engine with.

        Returns:
            queries (set, str): Unique tokens that are similar to the queried token and do not belong into the extremes.

        """
        sim_tokens = self.similar_tokens(token, sim_tokens)
        # print(sim_tokens)
        token_ids = self.word2ids(sim_tokens)
        high_ids = self.high_idf_ids(token_ids, bottom_lim, upper_lim)
        queries = [self.tfidf_id2word(id_) for id_ in high_ids]
        queries.append(token)
        return set(queries)

class Clio0(WordQueries):
    """Information retrieval based on word embeddings."""
    def __init__(self, dataframe, documents, word2vec):
        """Args:
            tfidf_model: Instance of the Tfidf class.
            word_queries: Instance of the WordQueries class.

        """
        self.tfidf_model = Tfidf(documents)
        self.word_queries = WordQueries(word2vec, documents)
        self.dataframe = dataframe
    
    def search(self, token=None, custom_list=None, sim_tokens=25, bottom_lim=35, upper_lim=95):
        """Wrapper function around the GtrSearch class.

        Args:
            word (str): Query term to search the dataset with.

        Returns:
            Subset of the dataframe with unique projects that are relevant to the query term.

        """
        if token:
            queries = self.word_queries.query_word(token, sim_tokens, bottom_lim, upper_lim)
        else:
            queries = set(custom_list)

        doc_intersection = {i:len(queries.intersection(doc)) for i, doc in enumerate(self.tfidf_model.documents) if len(queries.intersection(doc)) >= 1}
        sorted_idx = sorted(doc_intersection, key=lambda k: doc_intersection[k], reverse=True)
        return queries, self.dataframe.iloc[sorted_idx, :]