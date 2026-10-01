import pytest
import torch

import comfy_kitchen as ck

requires_int8_attention = pytest.mark.skipif(
    not torch.cuda.is_available()
    or getattr(torch.version, "hip", None)
    or not ck.int8_attention_is_available(),
    reason="requires the CUDA INT8 attention kernel",
)


def _operands(length, q_heads, kv_heads, head_dim, key_offset, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, q_heads, length, head_dim, device="cuda", dtype=torch.float16, generator=generator)
    k = torch.randn(1, kv_heads, length, head_dim, device="cuda", dtype=torch.float16, generator=generator)
    v = torch.randn(1, kv_heads, length, head_dim, device="cuda", dtype=torch.float16, generator=generator)
    if key_offset:
        # A component shared by every key is what makes the quantizer centre K on
        # an anchor key, which shifts the log-sum-exp.
        k = k + key_offset * torch.randn(
            1, kv_heads, 1, head_dim, device="cuda", dtype=torch.float16, generator=generator
        )
    return q, k, v


def _reference_lse(q, k, scale):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    rows = []
    for start in range(0, q.shape[2], 2048):
        scores = torch.einsum("bhqd,bhkd->bhqk", q[:, :, start:start + 2048].float(), k.float())
        rows.append(torch.logsumexp(scores * scale, dim=-1))
    return torch.cat(rows, dim=2)


def _merge(parts):
    lses = torch.stack([lse for _, lse in parts])
    total = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - total)
    return sum(weights[i][..., None] * parts[i][0].float() for i in range(len(parts))), total


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@requires_int8_attention
@pytest.mark.parametrize(
    ("length", "q_heads", "kv_heads", "head_dim", "key_offset"),
    [
        (4096, 4, 4, 128, 0.0),
        (4096, 4, 4, 128, 3.0),
        (4096, 8, 2, 128, 3.0),
        (4096, 4, 4, 64, 3.0),
        (2048, 2, 2, 96, 3.0),
    ],
)
def test_int8_attention_with_lse(length, q_heads, kv_heads, head_dim, key_offset):
    q, k, v = _operands(length, q_heads, kv_heads, head_dim, key_offset)
    scale = head_dim**-0.5

    output, lse = ck.int8_attention_with_lse(q, k, v)

    # Returning the log-sum-exp must not change the attention output.
    assert torch.equal(output, ck.int8_attention(q, k, v))
    assert lse.shape == (1, q_heads, length) and lse.dtype == torch.float32

    reference = _reference_lse(q, k, scale)
    tolerance = 0.01 if key_offset == 0.0 else 0.06
    assert (lse - reference).abs().max().item() < tolerance

    # Merging key shards by their log-sum-exp must reproduce attention over all keys.
    expanded = q_heads // kv_heads
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(expanded, dim=1), v.repeat_interleave(expanded, dim=1)
    )
    for shards in (2, 4):
        parts = [
            ck.int8_attention_with_lse(q, key_shard, value_shard)
            for key_shard, value_shard in zip(k.chunk(shards, dim=2), v.chunk(shards, dim=2))
        ]
        merged, merged_lse = _merge(parts)
        assert _cosine(merged, sdpa) > 0.9995
        assert (merged_lse - reference).abs().max().item() < tolerance
