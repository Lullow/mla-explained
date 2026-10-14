"""The claims of mla_demo.py as tests."""

import numpy as np
import pytest

from mla_demo import (DEEPSEEK_V2, Dims, cache_per_token_per_layer, mla_absorbed,
                      mla_unpack, multiply_adds_per_token)


def random_setup(dims, T, seed):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((T, dims.d))
    W_DKV = rng.standard_normal((dims.d_c, dims.d)) / dims.d ** 0.5
    W_UK = rng.standard_normal((dims.n_h, dims.d_h, dims.d_c)) / dims.d_c ** 0.5
    W_UV = rng.standard_normal((dims.n_h, dims.d_h, dims.d_c)) / dims.d_c ** 0.5
    W_Q = rng.standard_normal((dims.n_h, dims.d_h, dims.d)) / dims.d ** 0.5
    C = h @ W_DKV.T
    q = W_Q @ h[-1]
    return q, C, W_UK, W_UV


def test_toy_example_scores_and_outputs_agree():
    H = np.array([[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [0, 2, 1, 0]], float)
    W_DKV = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], float)
    W_Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], float)
    W_UK = np.array([[[1, 1], [0, 1]]], float)
    W_UV = np.array([[[1, 0], [1, 1]]], float)
    C = H @ W_DKV.T
    q = (W_Q @ H[-1])[None, :]

    assert C.tolist() == [[3, 1], [1, 1], [2, 2], [1, 2]]
    assert ((C @ W_UK[0].T) @ q[0]).tolist() == [6, 4, 8, 7]
    assert (C @ (W_UK[0].T @ q[0])).tolist() == [6, 4, 8, 7]
    a = mla_unpack(q, C, W_UK, W_UV, scale=1.0)
    b = mla_absorbed(q, C, W_UK, W_UV, scale=1.0)
    assert np.allclose(a, b)
    assert np.allclose(a[0], [1.835, 3.734], atol=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_paths_agree_small_random(seed):
    dims = Dims(d=64, n_h=8, d_h=16, d_c=32, d_rope=0)
    q, C, W_UK, W_UV = random_setup(dims, T=50, seed=seed)
    scale = 1 / dims.d_h ** 0.5
    assert np.allclose(mla_unpack(q, C, W_UK, W_UV, scale),
                       mla_absorbed(q, C, W_UK, W_UV, scale))


def test_paths_agree_deepseek_v2_sizes():
    q, C, W_UK, W_UV = random_setup(DEEPSEEK_V2, T=16, seed=1)
    scale = 1 / DEEPSEEK_V2.d_h ** 0.5
    assert C.shape == (16, 512)
    assert np.allclose(mla_unpack(q, C, W_UK, W_UV, scale),
                       mla_absorbed(q, C, W_UK, W_UV, scale))


def test_cache_sizes_match_the_paper():
    rows = cache_per_token_per_layer(DEEPSEEK_V2)
    by_prefix = {name.split(",")[0]: n for name, n in rows.items()}
    assert by_prefix["MHA"] == 32_768
    assert by_prefix["MLA"] == 576
    assert by_prefix["MHA"] / by_prefix["MLA"] == pytest.approx(56.9, abs=0.1)


def test_absorbed_path_does_not_grow_with_rebuilding():
    small = multiply_adds_per_token(DEEPSEEK_V2, T=1)
    big = multiply_adds_per_token(DEEPSEEK_V2, T=131_072)
    # Rebuilding keys and values costs d_c multiply-adds per cached number; attending costs one.
    assert big["MLA, unpack"] > 100 * big["MLA, absorbed"]
    assert big["MLA, absorbed"] == pytest.approx(DEEPSEEK_V2.d_c / DEEPSEEK_V2.d_h * big["MHA"], rel=0.01)
    assert small["MLA, absorbed"] > small["MHA"]
