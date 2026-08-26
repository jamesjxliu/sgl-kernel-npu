from enum import Enum
from typing import Optional, Sequence

import torch


class TransferDirection(Enum):
    H2D = 1
    D2H = 2


class TransferFlag(Enum):
    FAST2D = 2


def transfer_state_per_layer_direct_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    torch.ops.npu.transfer_state_per_layer_direct_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        flags.value,
    )


def transfer_state_all_layer_direct_lf_pf(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    torch.ops.npu.transfer_state_all_layer_direct_lf_pf(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        flags.value,
    )


def transfer_kv_dim_exchange(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    device_index_k_scale: Optional[torch.Tensor] = None,
    host_index_k_scale: Optional[torch.Tensor] = None,
    layer_start: int = 0,
    layer_num: int = -1,
    index_k_layer_start: Optional[int] = None,
    index_k_layer_num: Optional[int] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    """
    In the L1 and L2 radix cache scenarios, perform batch copy of KV data between the device and the host.

    Copies the layer range ``[layer_start, layer_start + layer_num)`` of each
    component; ``layer_num < 0`` means all layers.  Page indices are grouped
    into contiguous runs and each run is copied with
    ``min(run_length, layer_num)`` ``aclrtMemcpy2dAsync`` dispatches.

    Args:
        device_indices: token indices in device
        host_indices: token indices in host
        device_k: k_buffer in device
        host_k: k_buffer in host
        device_v: v_buffer in device
        host_v: v_buffer in host
        device_index_k: index_k_buffer in device
        host_index_k: index_k_buffer in host
        device_index_k_scale: per-token FP32 scale of the quantized Indexer in device
        host_index_k_scale: per-token FP32 scale of the quantized Indexer in host
        layer_start: first k/v layer to transfer (0-based, k/v layer space)
        layer_num: number of k/v layers to transfer; negative means all layers
        index_k_layer_start: first indexer slot to transfer.  ``None`` means
            the full indexer range (i.e. ``layer_start=0, layer_num=-1`` in
            the indexer slot space), matching the all-layer form.
        index_k_layer_num: number of indexer slots to transfer; ``None`` means
            all slots (only meaningful together with ``index_k_layer_start``).
        page_size: page size
        direction: only support H2D and D2H.
        flags: only FAST2D is supported, which indicates 2D data transfer via calling aclrtMemcpy2dAsync.
    """
    empty = torch.empty(0)
    torch.ops.npu.transfer_kv_dim_exchange(
        device_k,
        host_k,
        device_v,
        host_v,
        device_indices,
        host_indices,
        page_size,
        layer_start,
        layer_num,
        direction.value,
        flags.value,
    )
    if device_index_k is not None and host_index_k is not None:
        ik_start = 0 if index_k_layer_start is None else index_k_layer_start
        ik_num = -1 if index_k_layer_num is None else index_k_layer_num
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k,
            host_index_k,
            empty,
            empty,
            device_indices,
            host_indices,
            page_size,
            ik_start,
            ik_num,
            direction.value,
            flags.value,
        )
    if device_index_k_scale is not None and host_index_k_scale is not None:
        # Device scale cache is 4-D (layers, pages, page_size, 1) while the host
        # cache is 5-D (pages, layers, page_size, 1, 1); the kernel requires both
        # operands to be 5-D, so pad the device operand with a trailing singleton.
        if device_index_k_scale.dim() == 4:
            device_index_k_scale = device_index_k_scale.unsqueeze(-1)
        ik_start = 0 if index_k_layer_start is None else index_k_layer_start
        ik_num = -1 if index_k_layer_num is None else index_k_layer_num
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k_scale,
            host_index_k_scale,
            empty,
            empty,
            device_indices,
            host_indices,
            page_size,
            ik_start,
            ik_num,
            direction.value,
            flags.value,
        )


def transfer_kv_per_layer_dim_exchange(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    layer_id: int,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    device_index_k_scale: Optional[torch.Tensor] = None,
    host_index_k_scale: Optional[torch.Tensor] = None,
    index_k_layer_id: Optional[int] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    """Per-layer copy of KV data between device and host with contiguous-run merging.

    Thin single-layer form of :func:`transfer_kv_dim_exchange`: transfers layer
    ``layer_id`` of k/v and, when ``index_k_layer_id`` is not ``None``, slot
    ``index_k_layer_id`` of the indexer components.  Consecutive pages that
    advance in lock-step on both the device and host are merged into one
    ``aclrtMemcpy2dAsync`` call, reducing DMA dispatches from *num_pages* to
    *num_runs*.

    Args:
        layer_id: layer index in k/v's own layer space (0 .. num_layers-1).
        index_k_layer_id: layer index in the indexer slot space.  Must be
            provided together with ``device_index_k``/``host_index_k`` for the
            indexer components to be transferred; pass ``None`` to skip them
            for this layer (e.g. when the current layer is not an indexer
            layer).
    """
    transfer_kv_dim_exchange(
        device_indices=device_indices,
        host_indices=host_indices,
        device_k=device_k,
        host_k=host_k,
        device_v=device_v,
        host_v=host_v,
        device_index_k=device_index_k if index_k_layer_id is not None else None,
        host_index_k=host_index_k if index_k_layer_id is not None else None,
        device_index_k_scale=device_index_k_scale if index_k_layer_id is not None else None,
        host_index_k_scale=host_index_k_scale if index_k_layer_id is not None else None,
        layer_start=layer_id,
        layer_num=1,
        index_k_layer_start=index_k_layer_id,
        index_k_layer_num=1,
        page_size=page_size,
        direction=direction,
        flags=flags,
    )


def transfer_mamba_state(
    device_buf: torch.Tensor,
    host_buf: torch.Tensor,
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection = TransferDirection.H2D,
):
    """
    Transfer Mamba/SSM state between device (layer-first) and host (page-first).

    Device buffer layout: [num_layers, device_size, *state_shape]  (layer-first)
    Host buffer layout:   [host_size, num_layers, 1, *state_shape] (page-first)

    Uses aclrtMemcpy2dAsync for efficient 2D strided copy, transferring all
    layers for each slot index in a single 2D copy call.

    Args:
        device_buf: device Mamba state buffer [num_layers, size, *shape]
        host_buf: host Mamba state buffer [size, num_layers, 1, *shape]
        device_indices: slot indices in device buffer
        host_indices: slot indices in host buffer
        direction: H2D (host→device) or D2H (device→host)
    """
    torch.ops.npu.transfer_mamba_state(
        device_buf,
        host_buf,
        device_indices,
        host_indices,
        direction.value,
    )
