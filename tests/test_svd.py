import numpy as np
from trafficsign.math.matrix import Matrix
from trafficsign.math.svd import SvdDemonstrator


def test_svd_decomposition_and_reconstruction():
    # 4x3 matrix
    arr = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ]
    m = Matrix.from_array(arr)
    res = SvdDemonstrator.decompose_and_reconstruct(m, rank_k=2)

    assert res.rank_k == 2
    assert len(res.singular_values) == 3
    assert res.singular_values[0] >= res.singular_values[1] >= res.singular_values[2]
    assert res.energy_preserved_ratio > 0.99
    assert res.frobenius_error >= 0.0

    # Rank-3 should have approximately zero error
    res_full = SvdDemonstrator.decompose_and_reconstruct(m, rank_k=3)
    assert res_full.frobenius_error < 1e-10
