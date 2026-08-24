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
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
    layer_start: int = 0,
    layer_num: int = -1,
    index_k_layer_start: Optional[int] = None,
    index_k_layer_num: Optional[int] = None,
):
    """
    In the L1 and L2 radix cache scenarios, perform batch copy of KV data between the device and the host.

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
        page_size: page size
        direction: only support H2D and D2H.
        flags: only FAST2D is supported, which indicates 2D data transfer via calling aclrtMemcpy2dAsync.
        layer_start: first layer to transfer in the k/v layer space (dim 0 of
            device_k / dim 1 of host_k). Defaults to 0. Used for layer-group
            pipelining: a partial range makes the 2D copies cover only that
            group of layers, so per-layer completion events recorded between
            consecutive calls fire progressively.
        layer_num: number of k/v layers to transfer. Negative means all
            layers. Defaults to -1 (whole buffer, legacy behavior).
        index_k_layer_start: first layer in the index_k/scale layer space.
            Defaults to ``layer_start`` (identity mapping). DSA models store
            indexer K (and its FP32 scale) in a separate, smaller layer space
            (e.g. 21 indexer layers vs 78 total), so a partial k/v range must
            map the indexer layers into their own slot range.
        index_k_layer_num: number of index_k/scale layers to transfer.
            Defaults to ``layer_num``. 0 skips the index_k/scale copies (the
            k/v group contains no indexer layers).
    """
    torch.ops.npu.transfer_kv_dim_exchange(
        device_k,
        host_k,
        device_v,
        host_v,
        device_indices,
        host_indices,
        page_size,
        direction.value,
        flags.value,
        layer_start,
        layer_num,
    )
    if index_k_layer_start is None:
        index_k_layer_start = layer_start
    if index_k_layer_num is None:
        index_k_layer_num = layer_num
    if device_index_k is not None and host_index_k is not None and index_k_layer_num != 0:
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k,
            host_index_k,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
            index_k_layer_start,
            index_k_layer_num,
        )
    if device_index_k_scale is not None and host_index_k_scale is not None and index_k_layer_num != 0:
        # Device scale cache is 4-D (layers, pages, page_size, 1) while the host
        # cache is 5-D (pages, layers, page_size, 1, 1); the kernel requires both
        # operands to be 5-D, so pad the device operand with a trailing singleton.
        if device_index_k_scale.dim() == 4:
            device_index_k_scale = device_index_k_scale.unsqueeze(-1)
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k_scale,
            host_index_k_scale,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
            index_k_layer_start,
            index_k_layer_num,
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
