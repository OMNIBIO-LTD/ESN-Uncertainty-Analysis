"""
Echo State Network for UR5 state prediction.

The ESN is a recurrent neural network with FIXED random weights.
Only the readout layer is trained. This makes it:
  - Fast to train (closed-form solution)
  - Impossible to overfit the reservoir
  - Good at capturing temporal dynamics

How it works:
  1. Input (15-dim: state+action) is projected into a 512-dim reservoir
  2. Reservoir neurons are recurrently connected (sparse, random)
  3. At each timestep, reservoir state = mix of (old state + new input)
  4. The reservoir "echoes" past inputs — it has memory

The spectral radius controls how long echoes persist:
  - SR < 1.0 = stable, echoes decay (good for short-term prediction)
  - SR ≈ 1.0 = edge of chaos, long memory
  - SR > 1.0 = unstable, echoes grow (bad)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class EchoStateNetwork:
    """
    Echo State Network with sparse reservoir.

    Args:
        input_dim: Dimension of input vector (15 for UR5: state+action).
        reservoir_size: Number of reservoir neurons (default 512).
        spectral_radius: Controls echo memory length (default 0.95).
        input_scaling: Scale of input-to-reservoir weights (default 0.3).
        leaking_rate: How fast old state decays (0=keep all, 1=replace all).
        sparsity: Fraction of non-zero reservoir connections (default 0.1).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 512,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.3,
        leaking_rate: float = 0.3,
        sparsity: float = 0.1,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        rng = np.random.RandomState(seed)

        # W_in: projects input into reservoir space
        self.W_in = rng.uniform(
            -input_scaling, input_scaling,
            size=(reservoir_size, input_dim),
        ).astype(np.float64)

        # W: sparse recurrent reservoir connections
        W = sp.random(
            reservoir_size, reservoir_size,
            density=sparsity, format="csr",
            random_state=rng, dtype=np.float64,
        )
        W.data -= 0.5  # center around zero

        # Scale to desired spectral radius
        if W.nnz > 0:
            sr = self._compute_spectral_radius(W, rng)
            if sr > 0:
                W = W * (spectral_radius / sr)

        self.W = W.tocsr().astype(np.float64)

        # Internal state
        self.x = np.zeros(reservoir_size, dtype=np.float64)
        self._buf_input = np.zeros(reservoir_size, dtype=np.float64)
        self._buf_state = np.zeros(reservoir_size, dtype=np.float64)

    def update(self, u: np.ndarray) -> np.ndarray:
        """
        Process one timestep.

        Args:
            u: Input vector, shape (input_dim,).

        Returns:
            Copy of new reservoir state, shape (reservoir_size,).
        """
        alpha = self.leaking_rate
        u64 = u.astype(np.float64) if u.dtype != np.float64 else u

        # new_state = tanh(W_in @ u + W @ x)
        np.dot(self.W_in, u64, out=self._buf_input)
        self._buf_state[:] = self.W.dot(self.x)
        self._buf_state += self._buf_input
        np.tanh(self._buf_state, out=self._buf_state)

        # Leaky integration: x = (1-alpha)*x_old + alpha*new_state
        self._buf_state *= alpha
        self.x *= (1 - alpha)
        self.x += self._buf_state

        return self.x.copy()

    def reset(self) -> None:
        """Reset reservoir state to zeros (call between episodes)."""
        self.x[:] = 0.0

    @staticmethod
    def _compute_spectral_radius(
        W: sp.spmatrix,
        rng: np.random.RandomState,
    ) -> float:
        """Compute largest eigenvalue magnitude."""
        try:
            v0 = rng.rand(W.shape[0])
            eigenvalues, _ = spla.eigs(W, k=1, which="LM", maxiter=5000, v0=v0)
            return float(np.abs(eigenvalues[0]))
        except spla.ArpackNoConvergence:
            # Fallback: power iteration
            b = rng.rand(W.shape[0])
            b /= np.linalg.norm(b)
            for _ in range(100):
                b_new = W.dot(b)
                norm = np.linalg.norm(b_new)
                if norm == 0:
                    return 0.0
                b = b_new / norm
            return float(np.linalg.norm(W.dot(b)))
