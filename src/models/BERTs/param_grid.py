mBERT_grid = {
    "tok_space": {
        "max_length": [256],
    },
    "learning_space": {
        "learning_rate": [2e-5],
        "num_train_epochs": [5],
        "weight_decay": [0.01],
    }
}



modernBERT_grid = {
    "tok_space": {
        "max_length": [128],
    },
    "learning_space": {
        "learning_rate": [1e-5],
        "num_train_epochs": [10],
        "weight_decay": [0.01],
    }
}

