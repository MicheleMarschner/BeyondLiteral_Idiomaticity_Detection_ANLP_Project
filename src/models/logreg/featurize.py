import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

def _tokenize(s: str):
    return _TOKEN_RE.findall(str(s).lower())


class TfidfWeightedWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Fit:
      - Word2Vec on tokenized training texts
      - TF-IDF on the same tokenization (for per-token weights)
    Transform:
      - TF-IDF weighted average of word vectors (fallback to mean if needed)
    """
    def __init__(
        self,
        vector_size=200,
        window=5,
        min_count=2,
        sg=1,
        negative=10,
        epochs=10,
        seed=42,
        workers=4,
        tfidf_min_df=2,
        tfidf_max_df=0.95,
        tfidf_norm=None,          # keep None to preserve raw tfidf magnitudes as weights
        fallback="mean",          # "mean" or "zeros"
        max_features=None,
    ):
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

        self.w2v_ = None
        self.tfidf_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        # ---- Fit Word2Vec
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

        # ---- Fit TF-IDF with the SAME tokenization
        self.tfidf_ = TfidfVectorizer(
            tokenizer=_tokenize,
            preprocessor=None,
            token_pattern=None,     # required when providing custom tokenizer
            lowercase=False,        # we already lowercase in _tokenize
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            norm=self.tfidf_norm,   # None keeps weights as-is
            max_features=self.max_features,
        )
        self.tfidf_.fit(X)
        self.feature_names_ = self.tfidf_.get_feature_names_out()

        return self

    def transform(self, X):
        if self.w2v_ is None or self.tfidf_ is None:
            raise RuntimeError("TfidfWeightedWord2VecVectorizer is not fitted.")

        W = self.w2v_.wv
        dim = self.vector_size

        # sparse (n_docs, n_vocab)
        X_tfidf = self.tfidf_.transform(X)

        out = np.zeros((X_tfidf.shape[0], dim), dtype=np.float32)

        for i in range(X_tfidf.shape[0]):
            row = X_tfidf.getrow(i)
            idxs = row.indices
            vals = row.data

            num = np.zeros(dim, dtype=np.float32)
            den = 0.0

            # iterate only over terms with non-zero tfidf in this doc
            for j, w in zip(idxs, vals):
                token = self.feature_names_[j]
                if token in W:
                    num += (w * W[token]).astype(np.float32)
                    den += float(w)

            if den > 0:
                out[i] = num / den
            else:
                # fallback if none of the tf-idf tokens had vectors (or empty doc)
                if self.fallback == "mean":
                    toks = _tokenize(X[i])
                    vecs = [W[t] for t in toks if t in W]
                    if vecs:
                        out[i] = np.vstack(vecs).mean(axis=0).astype(np.float32)
                    # else keep zeros
                # else: keep zeros

        return out