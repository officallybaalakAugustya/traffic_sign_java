"""
Multi-dimensional Tensor abstraction mirroring com.trafficsign.math.Tensor.
Supports:
1. Scalar Broadcasting
2. Row-Vector Bias Broadcasting: [Batch, Features] + [Features] or [Batch, Features] + [1, Features]
3. Elementwise operations, reshaping, slicing batches, and matrix conversions.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from trafficsign.math.matrix import Matrix


class Tensor:
    """
    Multi-dimensional tensor abstraction mirroring com.trafficsign.math.Tensor.
    """

    def __init__(self, shape: Sequence[int], data: Optional[Union[np.ndarray, Sequence[float]]] = None):
        if shape is None or len(shape) == 0:
            raise ValueError("Tensor rank must be at least 1")
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"All tensor dimensions must be strictly positive. Shape: {shape}")

        self._shape = tuple(int(d) for d in shape)
        self._size = int(np.prod(self._shape))
        self._strides = self._compute_strides(self._shape)

        if data is None:
            self._data = np.zeros(self._size, dtype=np.float64)
        else:
            arr = np.array(data, dtype=np.float64).flatten()
            if arr.size != self._size:
                raise ValueError(f"Data array length ({arr.size}) does not match tensor element count ({self._size})")
            self._data = np.copy(arr)

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @classmethod
    def from_flat(cls, shape: Sequence[int], flat_data: Union[Sequence[float], np.ndarray]) -> Tensor:
        return cls(shape, flat_data)

    @classmethod
    def from_matrix(cls, m: Matrix) -> Tensor:
        if m is None:
            raise ValueError("Matrix must not be null")
        return cls((m.rows(), m.cols()), m.to_flat())

    @classmethod
    def zeros(cls, *shape: int) -> Tensor:
        return cls(shape)

    def rank(self) -> int:
        return len(self._shape)

    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def size(self) -> int:
        return self._size

    def get(self, *indices: int) -> float:
        flat_idx = self._get_flat_index(indices)
        return float(self._data[flat_idx])

    def set(self, value: float, *indices: int) -> None:
        flat_idx = self._get_flat_index(indices)
        self._data[flat_idx] = float(value)

    def reshape(self, *new_shape: int) -> Tensor:
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        for dim in new_shape:
            if dim <= 0:
                raise ValueError(f"New dimensions must be positive: {new_shape}")
        new_size = int(np.prod(new_shape))
        if new_size != self._size:
            raise ValueError(
                f"Cannot reshape tensor of size {self._size} into shape {new_shape} (size {new_size})"
            )
        return Tensor(new_shape, self._data)

    def add(self, other: Tensor) -> Tensor:
        if other is None:
            raise ValueError("other tensor must not be null")
        if self._shape == other._shape:
            return Tensor(self._shape, self._data + other._data)

        # Supported Bias Broadcasting Case 1: [Batch, Features] + [Features]
        if self.rank() == 2 and other.rank() == 1 and other._shape[0] == self._shape[1]:
            batch, features = self._shape
            res = self._data.reshape((batch, features)) + other._data.reshape((1, features))
            return Tensor(self._shape, res.flatten())

        # Supported Bias Broadcasting Case 2: [Batch, Features] + [1, Features]
        if self.rank() == 2 and other.rank() == 2 and other._shape[0] == 1 and other._shape[1] == self._shape[1]:
            batch, features = self._shape
            res = self._data.reshape((batch, features)) + other._data.reshape((1, features))
            return Tensor(self._shape, res.flatten())

        raise ValueError(f"Unsupported broadcasting shapes: {self._shape} and {other._shape}")

    def subtract(self, other: Tensor) -> Tensor:
        if other is None:
            raise ValueError("other tensor must not be null")
        if self._shape != other._shape:
            raise ValueError(f"Tensor shape mismatch for subtraction: {self._shape} vs {other._shape}")
        return Tensor(self._shape, self._data - other._data)

    def multiply_scalar(self, scalar: float) -> Tensor:
        return Tensor(self._shape, self._data * scalar)

    def element_multiply(self, other: Tensor) -> Tensor:
        if other is None:
            raise ValueError("other tensor must not be null")
        if self._shape != other._shape:
            raise ValueError(f"Tensor shape mismatch for elementwise multiplication: {self._shape} vs {other._shape}")
        return Tensor(self._shape, self._data * other._data)

    def apply(self, func: Callable[[float], float]) -> Tensor:
        if func is None:
            raise ValueError("function must not be null")
        vfunc = np.vectorize(func, otypes=[np.float64])
        return Tensor(self._shape, vfunc(self._data))

    def dot(self, other: Tensor) -> float:
        if other is None:
            raise ValueError("other tensor must not be null")
        if self._size != other._size:
            raise ValueError(f"Tensor element count mismatch for dot product: {self._size} vs {other._size}")
        return float(np.dot(self._data, other._data))

    def slice_batch(self, start_batch: int, end_batch: int) -> Tensor:
        if self.rank() < 1:
            raise RuntimeError("Cannot slice tensor with rank < 1")
        if start_batch < 0 or end_batch > self._shape[0] or start_batch >= end_batch:
            raise IndexError(
                f"Invalid slice bounds: [{start_batch}, {end_batch}) for batch size {self._shape[0]}"
            )
        sliced_batch_size = end_batch - start_batch
        sliced_shape = (sliced_batch_size,) + self._shape[1:]
        elements_per_sample = self._size // self._shape[0]

        start_idx = start_batch * elements_per_sample
        end_idx = end_batch * elements_per_sample
        sliced_data = self._data[start_idx:end_idx]
        return Tensor(sliced_shape, sliced_data)

    def to_matrix(self) -> Matrix:
        if self.rank() != 2:
            raise RuntimeError(f"Tensor must be rank 2 to convert to Matrix. Current rank: {self.rank()}")
        return Matrix.from_flat(self._shape[0], self._shape[1], self._data)

    def to_flat(self) -> np.ndarray:
        return np.copy(self._data)

    def to_numpy(self) -> np.ndarray:
        return self._data.reshape(self._shape).copy()

    def _get_flat_index(self, indices: Sequence[int]) -> int:
        if len(indices) != len(self._shape):
            raise ValueError(f"Indices rank {len(indices)} does not match tensor rank {len(self._shape)}")
        flat_idx = 0
        for i, (idx, dim_size) in enumerate(zip(indices, self._shape)):
            if idx < 0 or idx >= dim_size:
                raise IndexError(f"Index {idx} out of bounds for axis {i} (size {dim_size})")
            flat_idx += idx * self._strides[i]
        return flat_idx

    def __add__(self, other: Tensor) -> Tensor:
        return self.add(other)

    def __sub__(self, other: Tensor) -> Tensor:
        return self.subtract(other)

    def __mul__(self, other: Union[float, int, Tensor]) -> Tensor:
        if isinstance(other, (float, int)):
            return self.multiply_scalar(float(other))
        elif isinstance(other, Tensor):
            return self.element_multiply(other)
        raise TypeError(f"Unsupported operand type for *: {type(other)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return False
        return self._shape == other._shape and np.allclose(self._data, other._data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape}, size={self._size})"
