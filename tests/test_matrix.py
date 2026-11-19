import pytest
import numpy as np
from trafficsign.math.matrix import Matrix


def test_matrix_creation_and_bounds():
    m = Matrix(2, 3)
    assert m.rows() == 2
    assert m.cols() == 3
    assert m.get(0, 0) == 0.0

    m.set(1, 2, 5.5)
    assert m.get(1, 2) == 5.5

    with pytest.raises(IndexError):
        m.get(2, 0)
    with pytest.raises(IndexError):
        m.get(0, 3)
    with pytest.raises(ValueError):
        Matrix(0, 5)


def test_matrix_operations():
    a = Matrix.from_array([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix.from_array([[5.0, 6.0], [7.0, 8.0]])

    # Add
    c = a.add(b)
    assert c.get(0, 0) == 6.0
    assert c.get(1, 1) == 12.0

    # Subtract
    d = b.subtract(a)
    assert d.get(0, 0) == 4.0
    assert d.get(1, 1) == 4.0

    # Scalar Multiply
    e = a.multiply_scalar(2.0)
    assert e.get(0, 1) == 4.0
    assert e.get(1, 0) == 6.0

    # Matrix Multiply
    f = a.multiply(b)
    # [1*5+2*7, 1*6+2*8] = [19, 22]
    # [3*5+4*7, 3*6+4*8] = [43, 50]
    assert f.get(0, 0) == 19.0
    assert f.get(0, 1) == 22.0
    assert f.get(1, 0) == 43.0
    assert f.get(1, 1) == 50.0

    # Transpose
    t = a.transpose()
    assert t.get(0, 1) == 3.0
    assert t.get(1, 0) == 2.0

    # Apply function
    applied = a.apply(lambda x: x * x)
    assert applied.get(0, 1) == 4.0
    assert applied.get(1, 1) == 16.0

    # Dot product
    dot_val = Matrix.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert dot_val == 32.0
