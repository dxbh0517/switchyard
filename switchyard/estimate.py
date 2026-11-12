"""Quantization-aware VRAM estimator (pure; no GPU, no HTTP).

Predicts resident MiB from parameter count, weight quant, context length and KV-cache
settings. The budget layer uses this when no measured or configured cost exists.

Shapes default to Llama-family GQA (typical on 8–24 GB cards). Pass n_layers /
n_kv_heads / head_dim when you know them.
"""
from __future__ import annotations

# bytes per parameter, including GGUF super-block scale where it matters
_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0,
    "int8": 1.0, "q8_0": 1.0625,
    "q6_k": 0.78125,
    "q5_k": 0.6875, "q5_0": 0.65625,
    "q4_k": 0.5625, "q4_0": 0.53125, "int4": 0.5,
    "q3_k": 0.4375, "q2_k": 0.34375,
}
_ALIASES = {
    "float32": "fp32", "f32": "fp32",
    "float16": "fp16", "f16": "fp16",
    "bfloat16": "bf16",
    "e4m3": "fp8", "e5m2": "fp8",
    "8bit": "int8", "q8": "q8_0", "q8_1": "q8_0",
    "4bit": "int4", "q4": "q4_k", "gptq": "int4", "awq": "int4", "nf4": "int4",
    "q6": "q6_k", "q3": "q3_k", "q2": "q2_k",
}

# (n_params, n_layers, hidden_size, n_kv_heads, head_dim) — Llama-ish GQA
_SHAPES = (
    (1_000_000_000, 16, 2048, 8, 64),
    (3_000_000_000, 26, 3072, 8, 128),
    (8_000_000_000, 32, 4096, 8, 128),
    (14_000_000_000, 40, 5120, 8, 128),
    (32_000_000_000, 60, 6656, 8, 128),
    (70_000_000_000, 80, 8192, 8, 128),
)

_CUDA_OVERHEAD_MB = 512  # driver / allocator / torch workspace on a consumer card
_ACTIVATION = 0.05       # extra vs weights+KV for activations and fragmentation


def quant_bytes(quant: str) -> float:
    q = _norm_quant(quant)
    if q not in _BYTES:
        raise ValueError(f"unknown quant {quant!r}")
    return _BYTES[q]


def _norm_quant(quant: str) -> str:
    q = str(quant).lower().replace("-", "_").strip()
    q = _ALIASES.get(q, q)
    for prefix in ("q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"):
        if q.startswith(prefix):
            return prefix
    return q


def _infer_shape(n_params: int) -> dict:
    best = min(_SHAPES, key=lambda row: abs(row[0] - n_params))
    return {"n_layers": best[1], "hidden_size": best[2], "n_kv_heads": best[3], "head_dim": best[4]}


def breakdown(n_params: int, quant: str = "fp16", context_length: int = 4096,
              n_layers: int | None = None, n_kv_heads: int | None = None,
              head_dim: int | None = None, hidden_size: int | None = None,
              kv_quant: str | None = None) -> dict:
    """Weights + KV cache + overhead, in MiB. n_params is the raw parameter count."""
    n_params = int(n_params)
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    inferred = _infer_shape(n_params)
    n_layers = int(n_layers or inferred["n_layers"])
    n_kv_heads = int(n_kv_heads or inferred["n_kv_heads"])
    if head_dim is not None:
        head_dim = int(head_dim)
    elif hidden_size:
        head_dim = int(hidden_size) // max(1, n_kv_heads)
    else:
        head_dim = int(inferred["head_dim"])
    context_length = int(context_length)
    w_b = n_params * quant_bytes(quant)
    kv_b = (2 * n_layers * n_kv_heads * head_dim * context_length * quant_bytes(kv_quant or "fp16"))
    weights_mb = w_b / 2**20
    kv_mb = kv_b / 2**20
    overhead_mb = _CUDA_OVERHEAD_MB + _ACTIVATION * (weights_mb + kv_mb)
    return {
        "weights_mb": weights_mb, "kv_mb": kv_mb, "overhead_mb": overhead_mb,
        "total_mb": weights_mb + kv_mb + overhead_mb,
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "quant": _norm_quant(quant), "kv_quant": _norm_quant(kv_quant or "fp16"),
        "context_length": context_length,
    }


def estimate_vram_mb(n_params: int, quant: str = "fp16", context_length: int = 4096,
                     n_layers: int | None = None, n_kv_heads: int | None = None,
                     head_dim: int | None = None, hidden_size: int | None = None,
                     kv_quant: str | None = None) -> int:
    """Predicted resident VRAM in MiB (rounded)."""
    return int(round(breakdown(
        n_params, quant=quant, context_length=context_length, n_layers=n_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, hidden_size=hidden_size,
        kv_quant=kv_quant)["total_mb"]))


def gpu_layers_that_fit(vram_mb: int, n_params: int, n_layers: int | None = None,
                        quant: str = "q4_k", context_length: int = 4096,
                        **kwargs) -> int:
    """How many transformer layers fit in vram_mb (llama.cpp-style partial offload).

    Returns 0..n_layers. Embeddings/output are folded into the per-layer share —
    good enough to decide "all GPU" vs "some layers on CPU" on a consumer card.
    """
    parts = breakdown(n_params, quant=quant, context_length=context_length,
                      n_layers=n_layers, **kwargs)
    n_layers = parts["n_layers"]
    if parts["total_mb"] <= vram_mb:
        return n_layers
    per = (parts["weights_mb"] + parts["kv_mb"]) / max(1, n_layers)
    if per <= 0:
        return n_layers
    n = int((vram_mb - _CUDA_OVERHEAD_MB) // per)
    return max(0, min(n_layers, n))


def estimate_from_engine(cfg: dict) -> int | None:
    """Engine dict -> estimate, or None if there is no parameter count to go on."""
    n = cfg.get("n_params") if cfg.get("n_params") not in (None, "") else cfg.get("params")
    if not n:
        return None
    kw = {}
    if cfg.get("quant"):
        kw["quant"] = cfg["quant"]
    ctx = cfg.get("context_length", cfg.get("context"))
    if ctx:
        kw["context_length"] = int(ctx)
    for k in ("n_layers", "n_kv_heads", "head_dim", "hidden_size"):
        if cfg.get(k) not in (None, ""):
            kw[k] = int(cfg[k])
    if cfg.get("kv_quant"):
        kw["kv_quant"] = cfg["kv_quant"]
    return estimate_vram_mb(int(float(n)), **kw)
