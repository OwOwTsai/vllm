# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL attention backend for RDNA4 (gfx120x, wave32).

Completely standalone attention backend.  triton_attn.py contains zero FlyDSL
code; all FlyDSL logic lives here.

The backend reuses TritonAttentionBackend/MetadataBuilder for KV-cache layout
and metadata (those are purely about shapes, not kernel dispatch), but
FlyDSLAttentionImpl has its own forward() that is independent of Triton's.

Forward flow
────────────
1. KV cache store   — triton_reshape_and_cache_flash (same as Triton; required
                      before any attention kernel reads the cache)
2. FlyDSL attention — maybe_flydsl_attention() tries the FlyDSL kernel.
                      Declines automatically when: stream is being graph-
                      captured, kv_cache is fp8, alibi/sinks/softcap/sliding-
                      window/mm_prefix are active, or q is not bf16.
3. Triton fallback  — unified_attention() runs unchanged if FlyDSL declined.

Selected via VLLM_USE_FLYDSL_FLASH_ATTN=1 (rocm.py picks FLYDSL_ATTN backend).
Optional granular opt-outs:
    VLLM_USE_FLYDSL_ATTN_PREFILL=0   fall back prefill → Triton
    VLLM_USE_FLYDSL_ATTN_DECODE=0    fall back decode  → Triton
    VLLM_FLYDSL_ATTN_DECODE_MODE=2d  or 3d (default)
    VLLM_FLYDSL_ATTN_CONFIG_DIR=<path>  optional, overrides built-in configs
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.flydsl_attention import maybe_flydsl_attention
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


class FlyDSLAttentionImpl(AttentionImpl):
    """Standalone attention impl: FlyDSL kernel with Triton fallback.

    Does not inherit from TritonAttentionImpl. forward() is self-contained.
    """

    def fused_output_quant_supported(self, quant_key):
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8StaticTensorSym,
        )
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap if logits_soft_cap is not None else 0
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()
        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                f"Sinks must have num_heads={num_heads} rows, got {sinks.shape}."
            )
        self.supports_quant_query_input = current_platform.is_cuda()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported "
                "for FlyDSLAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        # ── Encoder attention (no KV cache) ──────────────────────────────────
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            if self.kv_cache_dtype.startswith("fp8"):
                raise NotImplementedError(
                    "quantization is not supported for encoder attention"
                )
            context_attention_fwd(
                q=query[:num_actual_tokens],
                k=key[:num_actual_tokens],
                v=value[:num_actual_tokens],
                o=output[:num_actual_tokens],
                b_start_loc=attn_metadata.query_start_loc,
                b_seq_len=attn_metadata.seq_lens,
                max_input_len=attn_metadata.max_query_len,
                is_causal=False,
                softmax_scale=self.scale,
                sliding_window_q=self.sliding_window[0],
                sliding_window_k=self.sliding_window[1],
            )
            return output

        # ── Decoder / cross-attention: KV cache path ─────────────────────────
        key_cache, value_cache = kv_cache.unbind(1)

        # 1. Store current K/V into the paged KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            if self.kv_cache_dtype.startswith("fp8"):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor
        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])

        # 2. FlyDSL attention kernel.
        #    maybe_flydsl_attention() declines automatically when:
        #    - kv_cache_dtype is fp8
        #    - alibi / sinks / softcap / sliding-window / mm_prefix active
        #    - q.dtype != bfloat16
        #    FlyDSL kernels run on torch.cuda.current_stream() so CUDA graph
        #    capture is supported — no stream-capturing guard needed.
        if maybe_flydsl_attention(
            is_prefill=(max_seqlen_q > 1),
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sinks=self.sinks,
            softcap=self.logits_soft_cap,
            sliding_window=self.sliding_window,
            qq_bias=None,
            mm_prefix=mm_prefix_range_tensor,
            kv_cache_dtype=self.kv_cache_dtype,
            output_scale=output_scale,
        ):
            return output

        # 3. Triton fallback.
        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output


class FlyDSLAttentionBackend(TritonAttentionBackend):
    """TritonAttentionBackend variant that uses FlyDSLAttentionImpl."""

    @staticmethod
    def get_name() -> str:
        return "FLYDSL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[FlyDSLAttentionImpl]:
        return FlyDSLAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return TritonAttentionMetadataBuilder
