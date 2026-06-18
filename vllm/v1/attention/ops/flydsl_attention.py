# Thin vLLM wrapper around the vllm_flydsl attention dispatcher.
#
# All kernel logic lives in vllm_flydsl.attention (FlyDSL_Tune package).
# This file adds only the vLLM-specific platform guard (_flydsl_flash_attn_master_on)
# and re-exports the symbols that flydsl_attn.py imports.

import os

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Import guard: requires `pip install -e FlyDSL_Tune` (or installed wheel).
# ---------------------------------------------------------------------------
try:
    from vllm_flydsl.attention import (
        FLYDSL_ATTN_CONFIG_DIR,
        NUM_PAR_SOFTMAX_SEGMENTS,
        is_flydsl_attn_decode_enabled,
        is_flydsl_attn_prefill_enabled,
        make_3d_segm_buffers,
        maybe_flydsl_attention,
    )
    _FLYDSL_ATTN_AVAILABLE = True
except ImportError as _e:
    FLYDSL_ATTN_CONFIG_DIR = ""
    _FLYDSL_ATTN_AVAILABLE = False
    logger.debug("vllm_flydsl not installed (%s); FlyDSL attention unavailable.", _e)

    def maybe_flydsl_attention(**_kwargs) -> bool:  # type: ignore[misc]
        return False

    def is_flydsl_attn_prefill_enabled() -> bool:  # type: ignore[misc]
        return False

    def is_flydsl_attn_decode_enabled() -> bool:  # type: ignore[misc]
        return False


if _FLYDSL_ATTN_AVAILABLE:
    logger.info(
        "FlyDSL attention kernels loaded (opt-in via VLLM_USE_FLYDSL_FLASH_ATTN=1)."
    )


def _flydsl_flash_attn_master_on() -> bool:
    """Master switch — adds the vLLM ROCm platform guard on top of the env var."""
    return (
        _FLYDSL_ATTN_AVAILABLE
        and os.environ.get("VLLM_USE_FLYDSL_FLASH_ATTN", "0") == "1"
        and current_platform.is_rocm()
    )
