# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import triton
import triton.language as tl


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def pdl_supported() -> bool:
    """Programmatic dependent launch (`griddepcontrol`) needs SM90+."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@triton.jit
def _gdc_noop():
    pass


if not pdl_supported():
    # Before SM90 `griddepcontrol` does not assemble: the kernels' PDL waits and
    # launch hints become no-ops (kernels then simply serialize on the stream).
    tl.extra.cuda.gdc_wait = _gdc_noop
    tl.extra.cuda.gdc_launch_dependents = _gdc_noop
