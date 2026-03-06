from __future__ import annotations
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, diags
from collections import Counter

from typing import Dict, Sequence, Optional, Tuple, Any

_TOKEN_RE = re.compile(r"\b\w\w+\b", re.UNICODE)

def _tokenize(s: str) -> list[str]:
    """Lowercase text and tokenize into Unicode “word” tokens (letters/digits/_) of length >= 2"""
    return _TOKEN_RE.findall(str(s).lower())


class MyTfidfVectorizer:
    def __init__(self,
        ngrams: Tuple[int, int] = (1, 1),
        min_df: float | int = 1,
        max_df: float | int = 1.0,
        norm: Optional[str] = "l2",
        max_features: int | None = None,
        smooth_idf: bool = True,
        sublinear_tf: bool = False
    ) -> None:
        """Simple TF-IDF vectorizer (sparse output)"""

        self.ngrams = ngrams
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        self.max_features = max_features
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
       
        # attributes to be learned in fit():
        self.vocab_: Dict[str, int] = {}  # term -> column index
        self.idf_: Optional[np.ndarray] = None  # shape: (n_features,)
        self.df_: Dict[str, int] = {}     # term -> document frequency (counts)
        self.n_docs_: int = 0             # number of documents seen in fit()
        self.feature_names_: Optional[list[str]] = None  # index -> term

    
    def _generate_ngrams(self, tokens: list[str]) -> list[str]:
        """Create n-gram strings from a token list based on the set ngram range"""
        
        min_n, max_n = self.ngrams
        if not tokens or min_n > max_n:
            return []
        
        terms = []
        N = len(tokens)

        for i in range(min_n, max_n + 1):
            if i <= 0 or i > N:
                continue
            for j in range(N - i + 1):
                terms.append(" ".join(tokens[j : j + i]))

        return terms
    

    def _top_terms_by_df(self, terms: list[str], k: int) -> list[str]:
        """Return the k terms with highest document frequency"""
        ranked = sorted(terms, key=lambda t: (-self.df_[t], t))
        return ranked[:k]


    def fit(self, corpus: Sequence[str]) -> MyTfidfVectorizer:
        """Builds vocabulary and IDF from a corpus"""
        corpus = list(corpus)
        if not corpus:
            raise ValueError("fit() received an empty corpus.")

        self.df_.clear()
        self.vocab_.clear()
        self.idf_ = None
        self.feature_names_ = None
        self.n_docs_ = 0

        # preprocess and split into n-grams for each document
        for doc in corpus:
            self.n_docs_ += 1

            tokens = _tokenize(doc)
            terms = self._generate_ngrams(tokens)

            unique_terms = set(terms)  # doc frequency: count each term once per doc
            for term in unique_terms:
                self.df_[term] = self.df_.get(term, 0) + 1

        # apply min_df / max_df filtering
        N = self.n_docs_                # number of docs

        # convert min_df/max_df (int or proportion) into absolute document-count thresholds
        min_count = int(np.ceil(self.min_df * N)) if isinstance(self.min_df, float) else int(self.min_df)
        max_count = int(np.floor(self.max_df * N)) if isinstance(self.max_df, float) else int(self.max_df)

        min_count = max(min_count, 1)
        max_count = min(max_count, N)

        # only keep terms within minimal and maximal document frequency bounds
        kept_terms = []
        for term, df in self.df_.items():
            if min_count <= df <= max_count:
                kept_terms.append(term)

        # caps at max features
        if self.max_features is not None and len(kept_terms) > self.max_features:
            kept_terms = self._top_terms_by_df(kept_terms, self.max_features)

        kept_terms = sorted(kept_terms)
        self.feature_names_ = kept_terms

        # build vocab
        self.vocab_ = {term: i for i, term in enumerate(kept_terms)}
        
        # compute idf
        idf = np.empty(len(kept_terms), dtype=np.float64)

        for term, idx in self.vocab_.items():
            df_t = self.df_[term]                   # document frequency for term t
            if self.smooth_idf:                     # smooth_idf implemented in sklearn package: log((1+N)/(1+df)) + 1
                idf[idx] = np.log((1.0 + N) / (1.0 + df_t)) + 1.0   # avoid division by zero
            else:                                   # log(N / df_t) + 1 
                idf[idx] = np.log(N / df_t) + 1.0   # avoid division by zero

        self.idf_ = idf

        return self


    def fit_transform(self, corpus: Sequence[str]) -> csr_matrix:
        """Learn vocab and idf from corpus then transform corpus to matrix"""

        self.fit(corpus)
        return self.transform(corpus)


    def transform(self, corpus: Sequence[str]) -> csr_matrix:
        """Transform a corpus into a sparse TF-IDF document-term matrix"""

        if self.idf_ is None or not self.vocab_:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        
        row_idx = []
        col_idx = []
        data = []

        corpus = list(corpus)
        n_docs = len(corpus)
        n_features = len(self.vocab_)
        
        # compute tfidf for each term in vocab_
        for i, doc in enumerate(corpus):
            tokens = _tokenize(doc)
            terms = self._generate_ngrams(tokens)

            # compute term counts for terms that exist in vocab_
            counts = Counter(t for t in terms if t in self.vocab_)
            if not counts:
                continue

            for term, tf in counts.items():
                j = self.vocab_[term]

                # optional sublinear TF
                if self.sublinear_tf:
                    tf = 1.0 + np.log(tf)           # tf = 1 + log(tf) for tf > 0

                # compute tf-idf:
                tfidf = float(tf) * float(self.idf_[j])     # tfidf(term) = tf(term) * idf_[term_index]

                row_idx.append(i)       # document index i
                col_idx.append(j)       # term/feature index j from vocab_
                data.append(tfidf)      # TF-IDF value to store at X[i, j]

        # Store only non-zero TF-IDF entries (sparse), avoiding a huge mostly-zero dense matrix.
        X = csr_matrix((data, (row_idx, col_idx)), shape=(n_docs, n_features), dtype=np.float64)            # CSR matrix from scipy

        # Normalize rows if norm == "l2"
        if self.norm == "l2":
            row_sq_sum = np.asarray(X.power(2).sum(axis=1)).ravel()     # sparse-friendly computation: for each row vector v do v = v / (||v||_2 + eps)
            norms = np.sqrt(row_sq_sum)
            norms[norms == 0.0] = 1.0               # avoid division by zero
            X = diags(1.0 / norms) @ X

        return X        # CSR matrix


    def get_feature_names_out(self) -> np.ndarray:
        """Return feature names ordered by column index"""

        if self.feature_names_ is None:
            raise ValueError("Vectorizer is not fitted. Call fit() first.")
        
        return np.array(self.feature_names_, dtype=object)


class TfidfWeightedWord2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size=200,
        window=5,
        min_count=2,
        sg=1,
        negative=10,
        epochs=10,
        seed=42,
        workers=1,                # deterministic behavior
        tfidf_min_df=2,
        tfidf_max_df=0.95,
        tfidf_norm=None,          # None preserves raw tfidf magnitudes as weights
        fallback="mean",          # "mean" or "zeros"
        smooth_idf=True,
        sublinear_tf=False,
        max_features: int | None = None,
    ):
        """Word2Vec pooled with TF-IDF token weights"""

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.seed = seed
        self.workers = workers

        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_norm = tfidf_norm
        self.fallback = fallback
        self.max_features = max_features
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.w2v_ = None
        self.tfidf_ = None
        self.feature_names_ = None

    def fit(self, X: Sequence[str]) -> TfidfWeightedWord2VecVectorizer:
        """Train Word2Vec and fit token-weight TF-IDF"""

        # Fit Word2Vec
        sentences = [_tokenize(x) for x in X]
        
        self.w2v_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            seed=self.seed,
            workers=self.workers,
        )
        self.w2v_.train(sentences, total_examples=len(sentences), epochs=self.epochs)

        # Fit TF-IDF with the SAME tokenization
        self.tfidf_ = MyTfidfVectorizer(
            ngrams=(1, 1),                 # for TF-IDF weights per token
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            norm=self.tfidf_norm,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features
        )
        self.tfidf_.fit(X)
        self.feature_names_ = self.tfidf_.get_feature_names_out()

        return self
    

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)


    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Convert texts into dense sentence embeddings using TF-IDF weighted Word2Vec pooling"""

        if self.w2v_ is None or self.tfidf_ is None:
            raise RuntimeError("TfidfWeightedWord2VecVectorizer is not fitted.")

        X_list = list(X)
        W = self.w2v_.wv
        dim = self.vector_size

        X_tfidf = self.tfidf_.transform(X_list)      # sparse (n_docs, n_vocab) (CSR)
        out = np.zeros((X_tfidf.shape[0], dim), dtype=np.float32) # dense sentence embeddings

        # compute TF-IDF-weighted average of Word2Vec embeddings per document
        for i in range(X_tfidf.shape[0]):
            row = X_tfidf.getrow(i)     # only iterate over non-zero TF-IDF entries for this document
            idxs = row.indices          # token column indices
            vals = row.data             # corresponding TF-IDF weights

            num = np.zeros(dim, dtype=np.float32)
            den = 0.0

            for j, w in zip(idxs, vals):            
                token = self.feature_names_[j]
                if token in W:            # skip OOV tokens
                    num += (w * W[token]).astype(np.float32)
                    den += float(w)

            if den > 0:
                out[i] = num / den      # compute TF-IDF-weighted average of embeddings
            else:
                # fallback if none of the tf-idf tokens has an embedding (or empty doc)
                if self.fallback == "mean":
                    toks = _tokenize(X_list[i])
                    vecs = [W[t] for t in toks if t in W]
                    if vecs:
                        out[i] = np.vstack(vecs).mean(axis=0).astype(np.float32)

        return out
    

def build_featurizer(model_family: str, params: Dict[str, Any]):
    """Create and configure the text featurizer for the selected model family"""

    print(params.get("max_df", 0.95))

    if model_family == "logreg_word2vec":
        return TfidfWeightedWord2VecVectorizer(
            # Word2Vec training hyperparameters
            vector_size=params.get("vector_size", 200),
            window=params.get("window", 5),
            min_count=params.get("min_count", 2),
            sg=1,
            negative=params.get("negative", 10),
            epochs=params.get("epochs", 10),
            workers=1,

            # TF-IDF weighting hyperparameters (for pooling)
            tfidf_min_df=params.get("min_df", 2),
            tfidf_max_df=params.get("max_df", 0.95),
            tfidf_norm=None,                 # keep None for weighting
            max_features=params.get("max_features", None),
            fallback=params.get("fallback", "mean"),
            smooth_idf=params.get("smooth_idf", True),
            sublinear_tf=params.get("sublinear_tf", False),
        )
    elif model_family == "logreg_tfidf":
        return MyTfidfVectorizer(
            ngrams=params.get("ngrams", (1, 2)),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            norm=params.get("norm", "l2"),
            max_features=params.get("max_features", None),
            smooth_idf=params.get("smooth_idf", True),
            sublinear_tf=params.get("sublinear_tf", False),
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")