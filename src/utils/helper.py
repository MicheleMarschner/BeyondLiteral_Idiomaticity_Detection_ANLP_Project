import os
import random
import numpy as np
import torch

def set_seeds(seed: int=51, deterministic: bool=True) -> None:
    """
    Sets seeds for complete reproducibility across all libraries and operations.

    Args:
        seed (int): Random seed value
    """

    # Python hashing (affects iteration order in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if deterministic:
            # cuDNN determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # CUDA matmul determinism (PyTorch recommends setting this env var)
            # Only needed for some CUDA versions/ops; harmless otherwise.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if deterministic:
        # Force deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(str(e))


    print(f"All random seeds set to {seed} for reproducibility")


