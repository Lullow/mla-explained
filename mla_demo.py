"""Multi-Head Latent Attention (MLA): the KV cache shrinks, the answer does not change.

The idea in one line: instead of caching every head's keys and values for every
token, cache one small latent vector per token (a "card") and let each head read
its own keys and values out of it. The trick that makes this free at inference is
that the reading step can be moved onto the query, so keys and values are never
actually rebuilt.

This file shows that with two code paths over the same cache:

  unpack    rebuild each head's keys and values from the cards, then attend
  absorbed  translate the query into card space once, attend to the cards directly

They give the same output. The file then counts what each attention variant stores
per token, and how many multiply-adds each path spends per generated token.

Not simulated: the decoupled RoPE key (the position stamp). It is counted in the
cache size, because DeepSeek-V2 stores it, but it is a separate mechanism from the
trick shown here. See README.md.

Run:  python mla_demo.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dims:
    d: int       # hidden size: the vector that represents a token inside the model
    n_h: int     # number of attention heads
    d_h: int     # size of one head's key, query or value
    d_c: int     # size of the latent vector, the card that MLA caches per token
    d_rope: int  # size of the decoupled RoPE key, also cached per token (not simulated)


DEEPSEEK_V2 = Dims(d=5120, n_h=128, d_h=128, d_c=512, d_rope=64)  # arXiv 2405.04434
LAYERS = 60             # DeepSeek-V2 has 60 transformer layers, each with its own cache
BYTES_PER_NUMBER = 2    # bf16
CONTEXT = 128 * 1024    # tokens


# --- the mechanism -----------------------------------------------------------

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def attend(q, K, V, scale):
    """One query against T cached keys. Returns the weighted blend of the values.

    q: (n,)   K: (T, n)   V: (T, m)   ->   (m,)
    """
    weights = softmax((K @ q) * scale)
    return weights @ V


def mla_unpack(q, C, W_UK, W_UV, scale):
    """Rebuild every head's keys and values from the cards, then attend.

    q:          (n_h, d_h)      the new token's query, one per head
    C:          (T, d_c)        the cache: one card per earlier token
    W_UK, W_UV: (n_h, d_h, d_c) each head's way of reading a key or a value out of a card
    """
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        K = C @ W_UK[i].T                  # (T, d_h)  this head's keys, rebuilt
        V = C @ W_UV[i].T                  # (T, d_h)  this head's values, rebuilt
        out[i] = attend(q[i], K, V, scale)
    return out


def mla_absorbed(q, C, W_UK, W_UV, scale):
    """Translate the query into card space, attend to the cards, unpack once.

    Same inputs and output as mla_unpack, but no key or value is ever built:
        q . (W_UK c)  ==  (W_UK^T q) . c
        sum_t w_t (W_UV c_t)  ==  W_UV (sum_t w_t c_t)
    """
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        q_card = W_UK[i].T @ q[i]            # (d_c,)  the query, translated once
        blend = attend(q_card, C, C, scale)  # (d_c,)  the cards serve as keys and values
        out[i] = W_UV[i] @ blend             # (d_h,)  read the value out of the blend
    return out


# --- the counts --------------------------------------------------------------

def cache_per_token_per_layer(dims, gqa_kv_heads=8):
    """Numbers stored per token in one layer, for four attention variants."""
    return {
        "MHA, every head its own K and V": 2 * dims.n_h * dims.d_h,
        f"GQA, {gqa_kv_heads} shared KV heads (Llama 3 70B)": 2 * gqa_kv_heads * dims.d_h,
        "MQA, one K and V shared by all heads": 2 * dims.d_h,
        f"MLA, one card of {dims.d_c} + RoPE key of {dims.d_rope}": dims.d_c + dims.d_rope,
    }


def multiply_adds_per_token(dims, T):
    """Multiply-adds one layer spends on attention for one new token, T tokens cached."""
    n_h, d_h, d_c = dims.n_h, dims.d_h, dims.d_c
    mha = T * n_h * d_h + T * n_h * d_h                      # scores + blend, K and V cached
    unpack = 2 * T * n_h * d_h * d_c + mha                   # rebuild K and V, then as MHA
    absorbed = (n_h * d_h * d_c                              # translate the query
                + T * n_h * d_c + T * n_h * d_c              # scores + blend, over cards
                + n_h * d_h * d_c)                           # read the value out of the blend
    return {"MHA": mha, "MLA, unpack": unpack, "MLA, absorbed": absorbed}


def fmt_bytes(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} PB"


# --- the runs ----------------------------------------------------------------

def toy_example():
    """The worked example: hidden size 4, one head of size 2, cards of size 2."""
    print("== Toy example: one head, four tokens, integer weights")
    H = np.array([[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [0, 2, 1, 0]], float)
    W_DKV = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], float)   # hidden -> card
    W_Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], float)     # hidden -> query
    W_UK = np.array([[[1, 1], [0, 1]]], float)               # (1 head, d_h, d_c): card -> key
    W_UV = np.array([[[1, 0], [1, 1]]], float)               # (1 head, d_h, d_c): card -> value

    C = H @ W_DKV.T                 # the cache: one card per token
    q = (W_Q @ H[-1])[None, :]      # the newest token asks; (1 head, d_h)
    print("cards                   ", C.astype(int).tolist())
    print("query                   ", q[0].astype(int).tolist())
    print("scores, keys rebuilt    ", ((C @ W_UK[0].T) @ q[0]).astype(int).tolist())
    print("scores, query translated", (C @ (W_UK[0].T @ q[0])).astype(int).tolist())
    a = mla_unpack(q, C, W_UK, W_UV, scale=1.0)
    b = mla_absorbed(q, C, W_UK, W_UV, scale=1.0)
    print("output, unpack          ", np.round(a[0], 3).tolist())
    print("output, absorbed        ", np.round(b[0], 3).tolist())
    assert np.allclose(a, b)
    print("same answer: yes")
    print()


def full_size_example(dims=DEEPSEEK_V2, T=32, seed=0):
    """DeepSeek-V2 sizes with random weights. One new token against T cached tokens."""
    print(f"== DeepSeek-V2 sizes, random weights: {dims.n_h} heads of {dims.d_h}, "
          f"cards of {dims.d_c}, hidden {dims.d}, {T} cached tokens")
    rng = np.random.default_rng(seed)
    f32 = np.float32
    h = rng.standard_normal((T, dims.d), dtype=f32)
    W_DKV = rng.standard_normal((dims.d_c, dims.d), dtype=f32) / dims.d ** 0.5
    W_UK = rng.standard_normal((dims.n_h, dims.d_h, dims.d_c), dtype=f32) / dims.d_c ** 0.5
    W_UV = rng.standard_normal((dims.n_h, dims.d_h, dims.d_c), dtype=f32) / dims.d_c ** 0.5
    W_Q = rng.standard_normal((dims.n_h, dims.d_h, dims.d), dtype=f32) / dims.d ** 0.5

    C = h @ W_DKV.T                      # (T, d_c)    the whole cache of this layer
    q = W_Q @ h[-1]                      # (n_h, d_h)  the newest token's query per head
    scale = 1 / dims.d_h ** 0.5
    a = mla_unpack(q, C, W_UK, W_UV, scale)
    b = mla_absorbed(q, C, W_UK, W_UV, scale)
    rebuilt = 2 * T * dims.n_h * dims.d_h
    print(f"cache for this layer:           {T} cards x {dims.d_c} = {C.size:,} numbers")
    print(f"keys and values the unpack path rebuilt from it: {rebuilt:,} numbers")
    print(f"max difference between the two outputs: {np.abs(a - b).max():.2e}")
    assert np.allclose(a, b, rtol=1e-3, atol=1e-4)
    print("same answer: yes")
    print()


def cache_table(dims=DEEPSEEK_V2):
    print(f"== What is cached per token ({LAYERS} layers, {BYTES_PER_NUMBER} bytes per number, "
          f"{CONTEXT // 1024}k context)")
    rows = cache_per_token_per_layer(dims)
    mla = next(n for name, n in rows.items() if name.startswith("MLA"))
    print(f"{'variant':<48} {'per token, layer':>16} {'per token, all':>14} "
          f"{'whole context':>14} {'vs MLA':>7}")
    for name, n in rows.items():
        per_token = n * LAYERS * BYTES_PER_NUMBER
        print(f"{name:<48} {n:>16,} {fmt_bytes(per_token):>14} "
              f"{fmt_bytes(per_token * CONTEXT):>14} {n / mla:>6.1f}x")
    print()


def work_table(dims=DEEPSEEK_V2, cached=(32, 4096, CONTEXT)):
    print("== Multiply-adds one layer spends on attention per generated token")
    names = list(multiply_adds_per_token(dims, 1))
    print(f"{'cached tokens':>14} " + " ".join(f"{n:>16}" for n in names))
    for T in cached:
        counts = multiply_adds_per_token(dims, T)
        print(f"{T:>14,} " + " ".join(f"{counts[n]:>16.2e}" for n in names))
    print("The unpack path pays for rebuilding keys and values on every token. "
          "The absorbed path pays a fixed translation plus dot products that are "
          f"{dims.d_c // dims.d_h}x longer than MHA's. Memory, not multiply-adds, is "
          "the bottleneck when generating, so MLA trades bytes for arithmetic.")
    print()


def main():
    toy_example()
    full_size_example()
    cache_table()
    work_table()


if __name__ == "__main__":
    main()
