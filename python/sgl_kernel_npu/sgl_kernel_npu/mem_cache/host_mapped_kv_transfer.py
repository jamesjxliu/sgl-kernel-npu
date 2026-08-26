"""Triton-ascend kernel for transferring KV pages between host-mapped NPU memory and HBM.

Demo 配套 host_mapped_kv.py：
- H2D: 从 host_npu_view (mapped dev_ptr, page-first layout) 搬到 device_k/v (layer-first layout)
- D2H: 反向

CANN 禁止在 mapped dev_ptr 上用 aclrtMemcpy*，所以这里用 triton-ascend kernel
直接 tl.load/tl.store 访问 mapped dev_ptr。triton-ascend 落到 AI core 上的
load/store 指令，符合 "Device computation" 的合法访问路径。

Layout:
- host_npu_view: [host_pages, layers, page_size, heads, head_dim]   (page-first)
- device_tensor: [layers, device_pages, page_size, heads, head_dim] (layer-first)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_mapped_transfer_kernel(
    host_ptr,  # npu_view (mapped dev_ptr), layout [host_pages, layers, ...]
    device_ptr,  # device_k/v on HBM, layout [layers, device_pages, ...]
    host_indices_ptr,  # token indices on NPU; host_indices[i*page_size] / page_size = host page index
    device_indices_ptr,
    host_page_stride,  # layers * page_payload
    host_layer_stride,  # page_payload
    device_layer_stride,  # num_device_pages * page_payload
    device_page_stride,  # page_payload
    page_payload,  # page_size * heads * head_dim
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECTION: tl.constexpr,  # 0=H2D (host→device), 1=D2H (device→host)
):
    page_i = tl.program_id(0)
    layer_i = tl.program_id(1)
    block_i = tl.program_id(2)

    # 读 page 索引：indices 存的是 token index，除以 PAGE_SIZE 得到 page index
    host_page_index = tl.load(host_indices_ptr + page_i * PAGE_SIZE) // PAGE_SIZE
    device_page_index = tl.load(device_indices_ptr + page_i * PAGE_SIZE) // PAGE_SIZE

    offsets = block_i * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < page_payload

    # page-first (host): page → layer → payload
    host_offset = host_page_index * host_page_stride + layer_i * host_layer_stride + offsets
    # layer-first (device): layer → page → payload
    device_offset = (
        layer_i * device_layer_stride + device_page_index * device_page_stride + offsets
    )

    if DIRECTION == 0:  # H2D: src=host mapped, dst=device HBM
        values = tl.load(host_ptr + host_offset, mask=mask, other=0)
        tl.store(device_ptr + device_offset, values, mask=mask)
    else:  # D2H: src=device HBM, dst=host mapped
        values = tl.load(device_ptr + device_offset, mask=mask, other=0)
        tl.store(host_ptr + host_offset, values, mask=mask)


def host_mapped_kv_transfer(
    host_npu_view: torch.Tensor,
    device_tensor: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    page_size: int,
    direction: int,
) -> None:
    """Transfer KV pages between host-mapped NPU memory and HBM via triton-ascend.

    Args:
        host_npu_view: NPU tensor backed by host memory (allocation.npu_view),
                       layout [host_pages, layers, page_size, heads, head_dim]
        device_tensor: NPU tensor on HBM, layout [layers, device_pages, page_size, heads, head_dim]
        host_indices: token indices in host (page_i*page_size 处存的是 host page 的起始 token index)
        device_indices: token indices in device
        page_size: page size
        direction: 0 = H2D (host→device), 1 = D2H (device→host)
    """
    if host_npu_view.dim() != 5:
        raise ValueError(
            f"host_npu_view must be 5-D [pages, layers, page_size, heads, head_dim], "
            f"got shape={tuple(host_npu_view.shape)}"
        )
    if device_tensor.dim() != 5:
        raise ValueError(
            f"device_tensor must be 5-D [layers, pages, page_size, heads, head_dim], "
            f"got shape={tuple(device_tensor.shape)}"
        )
    if host_npu_view.size(1) != device_tensor.size(0):
        raise ValueError(
            f"layer count mismatch: host={host_npu_view.size(1)}, device={device_tensor.size(0)}"
        )
    if host_npu_view.size(2) != page_size or device_tensor.size(2) != page_size:
        raise ValueError(
            f"page_size mismatch: host={host_npu_view.size(2)}, device={device_tensor.size(2)}, "
            f"expected={page_size}"
        )
    if host_npu_view.size(3) != device_tensor.size(3) or host_npu_view.size(4) != device_tensor.size(4):
        raise ValueError(
            f"heads/head_dim mismatch: host={tuple(host_npu_view.shape[3:])}, "
            f"device={tuple(device_tensor.shape[3:])}"
        )
    if direction not in (0, 1):
        raise ValueError(f"direction must be 0 (H2D) or 1 (D2H), got {direction}")

    num_pages = device_indices.size(0) // page_size
    if num_pages == 0:
        return

    num_layers = device_tensor.size(0)
    num_device_pages = device_tensor.size(1)
    heads = device_tensor.size(3)
    head_dim = device_tensor.size(4)
    page_payload = page_size * heads * head_dim

    # indices 必须在 NPU 上供 kernel 读取
    host_indices_npu = host_indices.to(device="npu", non_blocking=True)
    device_indices_npu = device_indices.to(device="npu", non_blocking=True)
    if host_indices_npu.dtype != torch.int64:
        host_indices_npu = host_indices_npu.to(torch.int64)
    if device_indices_npu.dtype != torch.int64:
        device_indices_npu = device_indices_npu.to(torch.int64)

    host_page_stride = num_layers * page_payload
    host_layer_stride = page_payload
    device_layer_stride = num_device_pages * page_payload
    device_page_stride = page_payload

    BLOCK = 2048
    grid = (num_pages, num_layers, triton.cdiv(page_payload, BLOCK))

    _kv_mapped_transfer_kernel[grid](
        host_npu_view,
        device_tensor,
        host_indices_npu,
        device_indices_npu,
        host_page_stride,
        host_layer_stride,
        device_layer_stride,
        device_page_stride,
        page_payload,
        PAGE_SIZE=page_size,
        BLOCK=BLOCK,
        DIRECTION=direction,
        num_warps=8,
    )
