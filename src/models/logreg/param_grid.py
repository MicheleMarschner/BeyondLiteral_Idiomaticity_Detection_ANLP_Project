
tfidf_param_grid = {
    # featurizer (MyTfidfVectorizer)
    "ngrams": [(1, 2)],
    "min_df": [2],
    "max_df": [0.9],
    "norm": ["l2"],
    "smooth_idf": [True],
    "sublinear_tf": [False],
    "max_features": [50000],

    # logreg
    "learning_rate": [0.1],
    "num_iterations": [5000],
    "lambda_reg": [0.0],
}

word2vec_param_grid = {
    # Word2Vec training
    "vector_size": [300],
    "window": [5],
    "negative": [5, 10],
    "min_count": [2],
    "epochs": [30],

    # TF-IDF weights for pooling
    "min_df": [5],
    "max_df": [0.95],
    "smooth_idf": [True],
    "sublinear_tf": [True],
    "max_features": [50000],

    # logreg
    "learning_rate": [0.01],
    "num_iterations": [5000],
    "lambda_reg": [1e-5],
    "norm": ['l2'],
}
