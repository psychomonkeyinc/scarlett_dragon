# dynamic_layers.py
# Utilities for dynamic/growable neural components (pure NumPy)
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

class GrowableLinear:
    """A growable linear layer (input_dim x output_dim) supporting runtime expansion.
    We keep weights as (in_dim, out_dim) like nn.Linear logic implemented in nn.py but without
    fatigue/plasticity. Used as an adapter when we must splice new capacity into existing models.
    """
    def __init__(self, input_dim:int, output_dim:int, activation=None, dtype=np.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        limit = math.sqrt(6.0/(input_dim+output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)
        self.biases = np.zeros((1, output_dim), dtype=dtype)
        self.activation = activation  # optional callable

    def forward(self, x:np.ndarray)->np.ndarray:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"GrowableLinear forward dim mismatch {x.shape[-1]} vs {self.input_dim}")
        y = x @ self.weights + self.biases
        if self.activation:
            y = self.activation(y)
        return y

    def expand_input(self, new_input_dim:int):
        if new_input_dim <= self.input_dim:
            return False
        add = new_input_dim - self.input_dim
        limit = math.sqrt(6.0/(new_input_dim + self.output_dim))
        extra = np.random.uniform(-limit, limit, (add, self.output_dim)).astype(self.dtype)
        self.weights = np.concatenate([self.weights, extra], axis=0)
        self.input_dim = new_input_dim
        return True

    def expand_output(self, new_output_dim:int):
        if new_output_dim <= self.output_dim:
            return False
        add = new_output_dim - self.output_dim
        limit = math.sqrt(6.0/(self.input_dim + new_output_dim))
        extra_w = np.random.uniform(-limit, limit, (self.input_dim, add)).astype(self.dtype)
        self.weights = np.concatenate([self.weights, extra_w], axis=1)
        extra_b = np.zeros((1, add), dtype=self.dtype)
        self.biases = np.concatenate([self.biases, extra_b], axis=1)
        self.output_dim = new_output_dim
        return True

class DimensionRegistry:
    """Central lightweight registry so modules can query and publish evolving dimensions.
    Not thread-safe; orchestrator mutates before/after cycle boundaries.
    """
    def __init__(self):
        self._dims = {}
    def publish(self, name:str, value:int):
        self._dims[name] = int(value)
    def get(self, name:str, default=None):
        return self._dims.get(name, default)
    def snapshot(self):
        return dict(self._dims)

# Utility functions for dimension handling

def pad_or_truncate(vec: np.ndarray, target: int) -> np.ndarray:
    """Return a 1D float32 vector of length target by zero-padding or truncating.
    Accepts None -> zeros.
    """
    if vec is None:
        return np.zeros(target, dtype=np.float32)
    v = np.array(vec, dtype=np.float32).reshape(-1)
    n = v.shape[0]
    if n == target:
        return v
    if n > target:
        return v[:target]
    out = np.zeros(target, dtype=np.float32)
    out[:n] = v
    return out

def ensure_2d(row_vec: np.ndarray) -> np.ndarray:
    v = np.array(row_vec, dtype=np.float32)
    if v.ndim == 1:
        return v.reshape(1, -1)
    return v

def param_count_linear(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)

def human(n: int) -> str:
    for unit in ['', 'K', 'M', 'B']:
        if n < 1000:
            return f"{n}{unit}"
        n //= 1000
    return f"{n}T"

# Global singleton (optional)
GLOBAL_DIM_REGISTRY = DimensionRegistry()
