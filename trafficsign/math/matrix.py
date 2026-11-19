"""
Pure Python/NumPy 2D Matrix implementation with strict dimension validation,
defensive copying, and linear algebra operations mirroring com.trafficsign.math.Matrix.
"""

from __future__ import annotations
import math
import random
from typing import Callable, List, Optional, Tuple, Union
import numpy as np


class Matrix:
    """
    2D Matrix implementation with strict dimension validation and linear algebra operations.
    """

    def __init__(self, rows: int, cols: int, data: Optional[Union[np.ndarray, List[List[float]]]] = None):
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Matrix dimensions must be strictly positive. Received: {rows}x{cols}")
        self._rows = rows
        self._cols = cols

        if data is None:
            self._data = np.zeros((rows, cols), dtype=np.float64)
        else:
            arr = np.array(data, dtype=np.float64)
            if arr.shape != (rows, cols):
                raise ValueError(f"Data shape {arr.shape} does not match specified ({rows}, {cols})")
            self._data = np.copy(arr)

    @classmethod
    def from_array(cls, values: Union[List[List[float]], np.ndarray]) -> Matrix:
        if values is None:
            raise ValueError("Initial values array must not be null")
        arr = np.array(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("Matrix cannot be initialized from empty or non-2D array")
        return cls(arr.shape[0], arr.shape[1], arr)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        return cls(rows, cols)

    @classmethod
    def from_flat(cls, rows: int, cols: int, flat: Union[List[float], np.ndarray]) -> Matrix:
        if flat is None:
            raise ValueError("flat array must not be null")
        arr = np.array(flat, dtype=np.float64)
        if arr.size != rows * cols:
            raise ValueError(f"Flat array length {arr.size} does not match matrix size {rows * cols}")
        return cls(rows, cols, arr.reshape((rows, cols)))

    @classmethod
    def random(cls, rows: int, cols: int, rng: Optional[random.Random] = None,
               min_val: float = 0.0, max_val: float = 1.0) -> Matrix:
        if rng is None:
            rng = random.Random()
        m = cls(rows, cols)
        range_val = max_val - min_val
        for r in range(rows):
            for c in range(cols):
                m._data[r, c] = min_val + rng.random() * range_val
        return m

    def rows(self) -> int:
        return self._rows

    def cols(self) -> int:
        return self._cols

    def get(self, r: int, c: int) -> float:
        self._check_bounds(r, c)
        return float(self._data[r, c])

    def set(self, r: int, c: int, val: float) -> None:
        self._check_bounds(r, c)
        self._data[r, c] = float(val)

    def add(self, other: Matrix) -> Matrix:
        if other is None:
            raise ValueError("other matrix must not be null")
        self._assert_same_shape(other)
        return Matrix(self._rows, self._cols, self._data + other._data)

    def subtract(self, other: Matrix) -> Matrix:
        if other is None:
            raise ValueError("other matrix must not be null")
        self._assert_same_shape(other)
        return Matrix(self._rows, self._cols, self._data - other._data)

    def multiply_scalar(self, scalar: float) -> Matrix:
        return Matrix(self._rows, self._cols, self._data * scalar)

    def multiply(self, other: Matrix) -> Matrix:
        if other is None:
            raise ValueError("other matrix must not be null")
        if self._cols != other._rows:
            raise ValueError(
                f"Incompatible dimensions for matrix multiplication: ({self._rows}x{self._cols}) * ({other._rows}x{other._cols})"
            )
        return Matrix(self._rows, other._cols, self._data @ other._data)

    def element_multiply(self, other: Matrix) -> Matrix:
        if other is None:
            raise ValueError("other matrix must not be null")
        self._assert_same_shape(other)
        return Matrix(self._rows, self._cols, self._data * other._data)

    def transpose(self) -> Matrix:
        return Matrix(self._cols, self._rows, self._data.T)

    @staticmethod
    def dot(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> float:
        if a is None or b is None:
            raise ValueError("vectors must not be null")
        arr_a = np.array(a, dtype=np.float64)
        arr_b = np.array(b, dtype=np.float64)
        if arr_a.shape != arr_b.shape:
            raise ValueError(f"Vectors must have identical length for dot product: {arr_a.size} vs {arr_b.size}")
        return float(np.dot(arr_a, arr_b))

    def apply(self, func: Callable[[float], float]) -> Matrix:
        if func is None:
            raise ValueError("mapping function must not be null")
        result = Matrix(self._rows, self._cols)
        vfunc = np.vectorize(func, otypes=[np.float64])
        result._data = vfunc(self._data)
        return result

    def sum(self) -> float:
        return float(np.sum(self._data))

    def sum_cols(self) -> np.ndarray:
        return np.sum(self._data, axis=0)

    def to_flat(self) -> np.ndarray:
        return self._data.flatten().copy()

    def to_array(self) -> np.ndarray:
        return np.copy(self._data)

    def _check_bounds(self, r: int, c: int) -> None:
        if r < 0 or r >= self._rows or c < 0 or c >= self._cols:
            raise IndexError(f"Matrix index ({r}, {c}) out of bounds for size ({self._rows}x{self._cols})")

    def _assert_same_shape(self, other: Matrix) -> None:
        if self._rows != other._rows or self._cols != other._cols:
            raise ValueError(
                f"Matrix shape mismatch: ({self._rows}x{self._cols}) vs ({other._rows}x{other._cols})"
            )

    def __add__(self, other: Matrix) -> Matrix:
        return self.add(other)

    def __sub__(self, other: Matrix) -> Matrix:
        return self.subtract(other)

    def __mul__(self, other: Union[float, int, Matrix]) -> Matrix:
        if isinstance(other, (float, int)):
            return self.multiply_scalar(float(other))
        elif isinstance(other, Matrix):
            return self.multiply(other)
        raise TypeError(f"Unsupported operand type for *: {type(other)}")

    def __matmul__(self, other: Matrix) -> Matrix:
        return self.multiply(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self._rows == other._rows and self._cols == other._cols and np.allclose(self._data, other._data)

    def __repr__(self) -> str:
        return f"Matrix({self._rows}x{self._cols})"

    def __str__(self) -> str:
        lines = [f"Matrix ({self._rows}x{self._cols}):"]
        for r in range(min(self._rows, 10)):
            row_vals = [f"{self._data[r, c]:8.4f}" for c in range(min(self._cols, 10))]
            if self._cols > 10:
                row_vals.append("...")
            lines.append(f"  [{', '.join(row_vals)}]")
        if self._rows > 10:
            lines.append("  ...")
        return "\n".join(lines)
