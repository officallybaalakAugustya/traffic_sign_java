import pytest
import numpy as np
from trafficsign.math.tensor import Tensor
from trafficsign.math.matrix import Matrix


def test_tensor_creation_and_indexing():
    t = Tensor((2, 3, 4))
    assert t.rank() == 3
    assert t.shape() == (2, 3, 4)
    assert t.size() == 24

    t.set(9.5, 1, 2, 3)
    assert t.get(1, 2, 3) == 9.5

    with pytest.raises(IndexError):
        t.get(2, 0, 0)


def test_tensor_broadcasting_and_arithmetic():
    # Shape [2, 3] + [3] bias vector
    batch = Tensor.from_flat((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bias = Tensor.from_flat((3,), [10.0, 20.0, 30.0])

    res = batch.add(bias)
    assert res.shape() == (2, 3)
    assert res.get(0, 0) == 11.0
    assert res.get(0, 1) == 22.0
    assert res.get(1, 2) == 36.0

    # Shape [2, 3] + [1, 3]
    bias2d = Tensor.from_flat((1, 3), [10.0, 20.0, 30.0])
    res2d = batch.add(bias2d)
    assert res2d.shape() == (2, 3)
    assert res2d.get(1, 0) == 14.0

    # Slicing
    sliced = batch.slice_batch(0, 1)
    assert sliced.shape() == (1, 3)
    assert sliced.get(0, 2) == 3.0

    # Reshape
    reshaped = batch.reshape(3, 2)
    assert reshaped.shape() == (3, 2)

    # To Matrix
    m = batch.to_matrix()
    assert isinstance(m, Matrix)
    assert m.rows() == 2
    assert m.cols() == 3
    assert m.get(1, 2) == 6.0
