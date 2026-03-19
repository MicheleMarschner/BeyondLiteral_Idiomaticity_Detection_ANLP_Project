mBERT_grid = {
    "tok_space": {
        "max_length": [128],
    },
    "learning_space": {
        "learning_rate": [2e-5],
        "num_train_epochs": [4],
        "weight_decay": [0.01],
        "batch_size": [8],
    }
}


modernBERT_grid = {
    "tok_space": {
        "max_length": [128, 256],
    },
    "learning_space": {
        "learning_rate": [1e-5, 2e-5],
        "num_train_epochs": [3, 4],
        "weight_decay": [0.01],
        "batch_size": [4, 8],
    }
}

