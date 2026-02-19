'''
param_grid = {
    "ngrams": [(1, 2), (2,2)],
    "min_df": [1, 2],
    "max_df": [0.95],
    "norm": ["l2"],
    "C": [0.5, 1.0, 2.0],
    "solver": ["liblinear"],     # keep simple
    "class_weight": [None, "balanced"],
    "penalty": ["l2"],           # start with l2 only
}'''

'''
"model_params": {
    "text_preprocessing": {
        "lowercase": true,
        "strip_accents": "unicode",
        "remove_punctuation": false,
        "remove_stopwords": false,
        "stem": false
    },

    "featurizer": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "norm": "l2"
    },

    "classifier": {
        "C": 1.0,
        "l1_ratio": 0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 2000
    }
  },
'''


tfidf_param_grid = {
    "ngrams": [(2,2)],
    "min_df": [1],
    "max_df": [0.95],
    "norm": ['l2'],
    "C": [1.0],
    "solver": ['liblinear'],     
    "class_weight": [None],
    "l1_ratio": [0],        
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
    "C": [1.0],
    "solver": ['liblinear'],     
    "class_weight": [None],
    "l1_ratio": [0],        
}

