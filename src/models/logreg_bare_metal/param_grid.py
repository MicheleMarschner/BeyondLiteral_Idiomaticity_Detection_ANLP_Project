tfidf_param_grid = {
    # TF-IDF featurizer
    "ngrams": [(1, 2), (2, 2)],
    "min_df": [1, 2, 5, 7, 10],
    "max_df": [0.8, 0.9, 0.95, 1.0],
    "norm": ["l2"],
    "learning_rate": [0.5, 0.1, 0.05, 0.01],
    "num_iterations": [2000, 5000, 7000],
    "lambda_reg": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    # "threshold": [0.3, 0.4, 0.5],
}

word2vec_param_grid = {
    "epochs": [10],
    "negative": [5],
    "window": [5],
    "vector_size": [100],
    "ngrams": [(2,2)],
    "min_df": [1],
    "max_df": [0.95],
    "norm": ['l2'],
    "learning_rate": [0.05],
    "num_iterations": [3000],
    "lambda_reg": [1e-3]      
}