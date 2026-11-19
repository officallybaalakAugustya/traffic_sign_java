"""
Singular Value Decomposition (SVD) engine and low-rank approximation demonstrator
mirroring com.trafficsign.math.SvdDemonstrator.
Uses scipy.linalg.svd / numpy.linalg.svd for numerically robust decomposition.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple
import numpy as np
import scipy.linalg
from trafficsign.math.matrix import Matrix


@dataclass(frozen=True)
class SvdResult:
    u: Matrix
    singular_values: np.ndarray
    v: Matrix
    reconstructed_rank_k: Matrix
    rank_k: int
    frobenius_error: float
    energy_preserved_ratio: float


class SvdDemonstrator:
    """
    SVD decomposition and low-rank approximation demonstrator.
    """

    @staticmethod
    def decompose_and_reconstruct(input_matrix: Matrix, rank_k: int) -> SvdResult:
        if input_matrix is None:
            raise ValueError("Input matrix must not be null")
        m = input_matrix.rows()
        n = input_matrix.cols()
        max_rank = min(m, n)

        if rank_k <= 0 or rank_k > max_rank:
            raise ValueError(f"Rank k ({rank_k}) must be in range [1, {max_rank}]")

        a = input_matrix.to_array()
        u, s, vt = scipy.linalg.svd(a, full_matrices=True)
        v = vt.T

        # Compute rank-k approximation: A_k = U[:, :k] @ diag(s[:k]) @ Vt[:k, :]
        u_k = u[:, :rank_k]
        s_k = np.diag(s[:rank_k])
        vt_k = vt[:rank_k, :]
        approx = u_k @ s_k @ vt_k

        # Frobenius error: ||A - A_k||_F
        diff = a - approx
        frobenius_error = float(np.linalg.norm(diff, 'fro'))

        # Energy preserved ratio
        total_energy = float(np.sum(s ** 2))
        approx_energy = float(np.sum(s[:rank_k] ** 2))
        energy_ratio = (approx_energy / total_energy) if total_energy > 0 else 1.0

        return SvdResult(
            u=Matrix.from_array(u),
            singular_values=s.copy(),
            v=Matrix.from_array(v),
            reconstructed_rank_k=Matrix.from_array(approx),
            rank_k=rank_k,
            frobenius_error=frobenius_error,
            energy_preserved_ratio=energy_ratio
        )
