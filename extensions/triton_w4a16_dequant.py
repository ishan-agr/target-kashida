"""Triton kernel: group-wise INT4 -> FP16 dequantisation (GPTQ/W4A16 layout).

STATUS: learning exercise written for the write-up, not used in production. In the
project every W4A16 kernel came from vLLM (awq_marlin / GPTQ exllama / Marlin) and
the one kernel failure I hit was in punica's `lora_shrink_op` (vLLM #28186). Writing
the dequant by hand is the fastest way to see *why* the ViT MLP (dim 4304) could
not be quantised with group_size=128: a group must tile K exactly.

Layout (GPTQ "act_order=False"):
  qweight : int32 [K // 8, N]   eight 4-bit values packed along K
  scales  : fp16  [K // G, N]
  zeros   : fp16  [K // G, N]   (unpacked here for clarity)
  W[k, n] = (q[k, n] - zeros[k // G, n]) * scales[k // G, n]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_kernel(q_ptr, s_ptr, z_ptr, out_ptr, K, N, G: tl.constexpr,
                    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    packed = tl.load(q_ptr + (offs_k[:, None] // 8) * N + offs_n[None, :], mask=mask, other=0)
    shift = (offs_k[:, None] % 8) * 4
    q = ((packed >> shift) & 0xF).to(tl.float16)

    g = offs_k[:, None] // G
    s = tl.load(s_ptr + g * N + offs_n[None, :], mask=mask, other=0.0)
    z = tl.load(z_ptr + g * N + offs_n[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + offs_k[:, None] * N + offs_n[None, :], (q - z) * s, mask=mask)


def dequant_w4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group: int) -> torch.Tensor:
    K, N = qweight.shape[0] * 8, qweight.shape[1]
    if K % group:
        raise ValueError(f"K={K} not divisible by group_size={group} (cf. Qwen3-VL ViT MLP 4304 % 128 = 80)")
    out = torch.empty((K, N), dtype=torch.float16, device=qweight.device)
    grid = (triton.cdiv(K, 64), triton.cdiv(N, 64))
    _dequant_kernel[grid](qweight, scales, zeros, out, K, N, G=group, BLOCK_K=64, BLOCK_N=64)
    return out


def quantize_ref(w: torch.Tensor, group: int):
    """Asymmetric round-to-nearest reference quantiser (what GPTQ improves on)."""
    K, N = w.shape
    wg = w.float().view(K // group, group, N)
    mn, mx = wg.amin(1), wg.amax(1)
    scale = ((mx - mn) / 15).clamp(min=1e-8)
    zero = (-mn / scale).round()
    q = (wg / scale[:, None] + zero[:, None]).round().clamp(0, 15).to(torch.int32).view(K, N)
    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= q[i::8] << (4 * i)
    return packed, scale.half(), zero.half()


if __name__ == "__main__":
    torch.manual_seed(0)
    K, N, G = 4096, 4096, 128
    w = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02
    qw, s, z = quantize_ref(w, G)
    w_hat = dequant_w4(qw, s, z, G)
    ref = ((torch.stack([(qw >> (4 * i)) & 0xF for i in range(8)], 1).view(K, N).half()
            - z.repeat_interleave(G, 0)) * s.repeat_interleave(G, 0))
    print("kernel == torch reference:", torch.allclose(w_hat, ref, atol=1e-3))
    print("relative RTN error ||W-W_hat||/||W||:", ((w - w_hat).norm() / w.norm()).item())
    print("ms/call:", triton.testing.do_bench(lambda: dequant_w4(qw, s, z, G)))
