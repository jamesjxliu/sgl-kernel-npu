import time
import unittest

import torch
from sgl_kernel_npu.kvcacheio import (
    TransferDirection,
    TransferFlag,
    transfer_kv_dim_exchange,
)

try:
    from sgl_kernel_npu.kvcacheio import transfer_kv_dim_exchange_table
except ImportError:
    transfer_kv_dim_exchange_table = None

try:
    from memfabric_hybrid import offload as _mf_offload
except ImportError:
    _mf_offload = None

# example comes from Qwen3-32B, TP=2
TP = 2
NUM_KV_HEADS = 8
NUM_LAYERS = 64
NUM_PAGES = 30
PAGE_SIZE = 128
HEAD_NUM_PER_TP = int(NUM_KV_HEADS / TP)
HEAD_DIM = 128


class TestTransferKV(unittest.TestCase):

    def _kv_transfer(
        self, direct: TransferDirection, v_empty: bool, index_k_empty: bool = True
    ):
        torch.npu.set_device(0)

        device_kv_buffer = torch.ones(
            (2, NUM_LAYERS, NUM_PAGES, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )
        device_k = device_kv_buffer[0]
        device_v = torch.empty(0) if v_empty else device_kv_buffer[1]

        host_kv_buffer = torch.zeros(
            (2, NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )

        self.assertNotEqual(
            device_kv_buffer.sum(),
            host_kv_buffer.sum(),
            "device value should not be equal to host value",
        )

        host_k = host_kv_buffer[0]
        host_v = torch.empty(0) if v_empty else host_kv_buffer[1]

        device_index_k = torch.ones(
            (NUM_LAYERS, NUM_PAGES, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )

        host_index_k = torch.zeros(
            (NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )

        device_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)

        stream = torch.npu.Stream()
        start = time.time()
        with torch.npu.stream(stream):
            if index_k_empty:
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_k,
                    host_k=host_k,
                    device_v=device_v,
                    host_v=host_v,
                    page_size=PAGE_SIZE,
                    direction=direct,
                )
            else:
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_k,
                    host_k=host_k,
                    device_v=device_v,
                    host_v=host_v,
                    device_index_k=device_index_k,
                    host_index_k=host_index_k,
                    page_size=PAGE_SIZE,
                    direction=direct,
                )

        end = time.time()
        direct_str = "D2H" if direct == TransferDirection.D2H else "H2D"
        copy_times = NUM_PAGES
        if v_empty is False:
            copy_times += NUM_PAGES
        if index_k_empty is False:
            copy_times += NUM_PAGES

        total_size = (
            copy_times
            * NUM_LAYERS
            * PAGE_SIZE
            * HEAD_NUM_PER_TP
            * HEAD_DIM
            * torch.bfloat16.itemsize
        )
        print(
            f"kv transfer {direct_str}, {v_empty=}, {index_k_empty=}, "
            f"2d copy times is {copy_times}, "
            f"total copy size is {total_size} bytes, "
            f"total duration {float((end - start) * 1000):.3f}ms"
        )
        torch.npu.synchronize()
        return device_kv_buffer, host_kv_buffer

    def _k_transfer(self, direct_str):
        return self._kv_transfer(direct_str, True)

    def test_kv_copy_d2h(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.D2H, False)

        self.assertAlmostEqual(
            host_kv.sum().item(),
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value should be equal to device value after transfer kv d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item(),
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() should be equal to numel() after transfer kv d2h",
        )

    def test_kv_copy_h2d(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.H2D, False)

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            host_kv.sum().item(),
            delta=1e-3,
            msg="device value should be equal to host value after transfer kv h2d",
        )

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device value sum() should be equal to 0 after transfer kv h2d",
        )

    def test_kv_index_k_copy_d2h(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.D2H, False, False)

        self.assertAlmostEqual(
            host_kv.sum().item(),
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value should be equal to device value after transfer kv and index k d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item(),
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() should be equal to numel() after transfer kv and index d2h",
        )

    def test_kv_index_k_copy_h2d(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.H2D, False, False)

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            host_kv.sum().item(),
            delta=1e-3,
            msg="device value should be equal to host value after transfer kv and index k h2d",
        )

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device value sum() should be equal to 0 after transfer kv and index k h2d",
        )

    def test_k_copy_d2h(self):
        device_kv, host_kv = self._k_transfer(TransferDirection.D2H)

        self.assertAlmostEqual(
            host_kv.sum().item() * 2,
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value * 2 should be equal to device value after transfer k d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item() * 2,
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() * 2 should be equal to numel() after transfer k d2h",
        )

    def test_k_copy_h2d(self):
        device_kv, host_kv = self._k_transfer(TransferDirection.H2D)

        self.assertAlmostEqual(
            device_kv[0].sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device k sum() should be equal to 0 after transfer k h2d",
        )

        self.assertAlmostEqual(
            device_kv[1].sum().cpu().item() * 2,
            host_kv.numel(),
            delta=1e-3,
            msg="device v sum() * 2 should be equal to host value after transfer k h2d",
        )

    def _scale_transfer(self, direct: TransferDirection):
        """Transfer only the quantized-Indexer FP32 scale cache.

        Device layout is (layers, pages, page_size, 1) FP32 (4-D, the wrapper
        appends the trailing singleton dim); host layout is page-first
        (pages, layers, page_size, 1, 1) FP32.
        """
        torch.npu.set_device(0)

        # Distinct value per (layer, page) so that a wrong page/layer stride or
        # offset -- the layout-sensitive part of this feature -- fails the
        # element-wise assertions below (uniform values would hide it).
        base = (
            torch.arange(NUM_LAYERS * NUM_PAGES, dtype=torch.float32)
            .view(NUM_LAYERS, NUM_PAGES)
            .add_(1.0)
        )

        device_scale = torch.zeros(
            (NUM_LAYERS, NUM_PAGES, PAGE_SIZE, 1),
            dtype=torch.float32,
            device="npu",
        )
        host_scale = torch.zeros(
            (NUM_PAGES, NUM_LAYERS, PAGE_SIZE, 1, 1),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

        # Non-empty dummy k/v slabs mirror the real H2D/D2H path, where the
        # k/v and scale buffers are distinct: only the scale operands below
        # move scale data (avoids the idempotent double write of the old test).
        device_kv = torch.zeros(
            (NUM_LAYERS, NUM_PAGES, PAGE_SIZE, 1, 1),
            dtype=torch.float32,
            device="npu",
        )
        host_kv = torch.zeros(
            (NUM_PAGES, NUM_LAYERS, PAGE_SIZE, 1, 1),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

        if direct == TransferDirection.D2H:
            device_scale.copy_(base.unsqueeze(-1).unsqueeze(-1))
        else:
            # Host layout is 5-D (pages, layers, page_size, 1, 1); copy_ aligns
            # trailing dims, so append 3 trailing singletons to the (pages, layers)
            # matrix to broadcast page_size (dim2: 1 -> page_size).
            host_scale.copy_(
                base.permute(1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )

        device_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)

        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_kv,
                host_k=host_kv,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                device_index_k_scale=device_scale,
                host_index_k_scale=host_scale,
                page_size=PAGE_SIZE,
                direction=direct,
            )
        torch.npu.synchronize()
        return device_scale, host_scale

    def test_scale_copy_d2h(self):
        device_scale, host_scale = self._scale_transfer(TransferDirection.D2H)

        # host_scale[page, layer, token, 0, 0] must equal
        # device_scale[layer, page, token, 0] for every page/layer.
        expected = (
            torch.arange(NUM_LAYERS * NUM_PAGES, dtype=torch.float32)
            .view(NUM_LAYERS, NUM_PAGES)
            .add_(1.0)
            .permute(1, 0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(NUM_PAGES, NUM_LAYERS, PAGE_SIZE, 1, 1)
        )
        self.assertTrue(
            torch.allclose(host_scale, expected, atol=1e-6),
            msg="host scale should equal the per-(page, layer) device values after d2h",
        )

    def test_scale_copy_h2d(self):
        device_scale, host_scale = self._scale_transfer(TransferDirection.H2D)

        # device_scale[layer, page, token, 0] must equal
        # host_scale[page, layer, token, 0, 0] for every page/layer.
        expected = (
            torch.arange(NUM_LAYERS * NUM_PAGES, dtype=torch.float32)
            .view(NUM_LAYERS, NUM_PAGES)
            .add_(1.0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(NUM_LAYERS, NUM_PAGES, PAGE_SIZE, 1)
        )
        self.assertTrue(
            torch.allclose(device_scale.cpu(), expected, atol=1e-6),
            msg="device scale should equal the per-(layer, page) host values after h2d",
        )


class TestTransferKVLayerRange(unittest.TestCase):
    """Layer-group transfers (layer_start/layer_num and index_k_* ranges).

    Each layer holds a distinct value so that a wrong layer offset, pitch or
    height fails the per-layer assertions below (uniform values would hide it).
    """

    NUM_LAYERS = 16
    NUM_INDEX_K_LAYERS = 6  # DSA-style: fewer indexer layers than total layers
    NUM_PAGES = 4
    PAGE_SIZE = 128
    HEADS = 1
    HEAD_DIM = 128

    def _make_device_buffer(self, num_layers, fill):
        buf = torch.zeros(
            (num_layers, self.NUM_PAGES, self.PAGE_SIZE, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )
        for layer in range(num_layers):
            buf[layer].fill_(fill(layer))
        return buf

    def _make_host_buffer(self, num_layers):
        return torch.zeros(
            (self.NUM_PAGES, num_layers, self.PAGE_SIZE, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )

    def _transfer(
        self,
        direct: TransferDirection,
        layer_start: int,
        layer_num: int,
        index_k_layer_start: int = None,
        index_k_layer_num: int = None,
    ):
        torch.npu.set_device(0)

        device_k = self._make_device_buffer(self.NUM_LAYERS, lambda l: l + 1)
        device_v = self._make_device_buffer(self.NUM_LAYERS, lambda l: 100 + l + 1)
        device_index_k = self._make_device_buffer(
            self.NUM_INDEX_K_LAYERS, lambda l: 200 + l + 1
        )
        host_k = self._make_host_buffer(self.NUM_LAYERS)
        host_v = self._make_host_buffer(self.NUM_LAYERS)
        host_index_k = self._make_host_buffer(self.NUM_INDEX_K_LAYERS)

        device_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        host_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )

        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=device_v,
                host_v=host_v,
                device_index_k=device_index_k,
                host_index_k=host_index_k,
                page_size=self.PAGE_SIZE,
                direction=direct,
                layer_start=layer_start,
                layer_num=layer_num,
                index_k_layer_start=index_k_layer_start,
                index_k_layer_num=index_k_layer_num,
            )
        torch.npu.synchronize()
        return (
            device_k,
            device_v,
            device_index_k,
            host_k,
            host_v,
            host_index_k,
        )

    def _check_d2h_range(self, host, num_layers, lo, hi, fill, base):
        # host is (pages, layers, ...); transpose checks into layer-major.
        host_layer_major = host.permute(1, 0, 2, 3, 4)
        for layer in range(num_layers):
            if lo <= layer < hi:
                self.assertTrue(
                    torch.all(
                        torch.eq(host_layer_major[layer].cpu(), fill(layer))
                    ),
                    f"layer {layer} should have been transferred",
                )
            else:
                self.assertTrue(
                    torch.all(
                        torch.eq(host_layer_major[layer].cpu(), base)
                    ),
                    f"layer {layer} should not have been transferred",
                )

    def test_layer_range_d2h(self):
        _, _, _, host_k, host_v, _ = self._transfer(
            TransferDirection.D2H, layer_start=3, layer_num=5
        )
        self._check_d2h_range(host_k, self.NUM_LAYERS, 3, 8, lambda l: l + 1, 0)
        self._check_d2h_range(host_v, self.NUM_LAYERS, 3, 8, lambda l: 100 + l + 1, 0)

    def test_layer_range_h2d(self):
        # Prefill host with per-layer values, wipe device, transfer a group
        # back and verify only that group moved.
        torch.npu.set_device(0)
        device_k = torch.zeros(
            (self.NUM_LAYERS, self.NUM_PAGES, self.PAGE_SIZE, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )
        host_k = self._make_host_buffer(self.NUM_LAYERS)
        for layer in range(self.NUM_LAYERS):
            host_k[:, layer].fill_(layer + 1)
        device_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        host_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        with torch.npu.stream(torch.npu.Stream()):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
                layer_start=5,
                layer_num=4,
            )
        torch.npu.synchronize()
        device_k_cpu = device_k.cpu()
        for layer in range(self.NUM_LAYERS):
            expected = layer + 1 if 5 <= layer < 9 else 0
            self.assertTrue(
                torch.all(torch.eq(device_k_cpu[layer], expected)),
                f"layer {layer} H2D content mismatch",
            )

    def test_index_k_layer_range_d2h(self):
        # k/v group [3, 8) while index_k uses its own slot range [2, 5),
        # mirroring DSA models where indexer layers live in a smaller space.
        _, _, _, host_k, host_v, host_index_k = self._transfer(
            TransferDirection.D2H,
            layer_start=3,
            layer_num=5,
            index_k_layer_start=2,
            index_k_layer_num=3,
        )
        self._check_d2h_range(host_k, self.NUM_LAYERS, 3, 8, lambda l: l + 1, 0)
        self._check_d2h_range(host_v, self.NUM_LAYERS, 3, 8, lambda l: 100 + l + 1, 0)
        self._check_d2h_range(
            host_index_k, self.NUM_INDEX_K_LAYERS, 2, 5, lambda l: 200 + l + 1, 0
        )

    def test_index_k_layer_num_zero_skips_index_k(self):
        # A k/v group containing no indexer layers: index_k must stay zero.
        _, _, _, host_k, _, host_index_k = self._transfer(
            TransferDirection.D2H,
            layer_start=3,
            layer_num=5,
            index_k_layer_start=0,
            index_k_layer_num=0,
        )
        self._check_d2h_range(host_k, self.NUM_LAYERS, 3, 8, lambda l: l + 1, 0)
        self.assertTrue(
            torch.all(torch.eq(host_index_k.cpu(), 0)),
            "index_k should not be transferred when index_k_layer_num=0",
        )

    def test_layer_range_out_of_bounds_rejected(self):
        torch.npu.set_device(0)
        device_k = self._make_device_buffer(self.NUM_LAYERS, lambda l: 1)
        host_k = self._make_host_buffer(self.NUM_LAYERS)
        device_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        host_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        with self.assertRaises(RuntimeError):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
                layer_start=12,
                layer_num=5,  # 12 + 5 > 16
            )


class TestTransferKVRunMerge(unittest.TestCase):
    """Contiguous page-run merging (full-layer, one-shot transfers).

    When host and device page indices are contiguous, the kernel merges the
    per-page 2D copies into per-layer 2D copies. These tests force both paths:
    long runs (>= layers) take the merged form, short runs fall back to the
    per-page form. Every (layer, page) holds a distinct value so a wrong
    page/layer pitch or run boundary fails the element-wise checks (uniform
    values would hide it), and untouched pages must stay zero (catches
    out-of-run writes from bad strides).
    """

    NUM_LAYERS = 16
    NUM_PAGES = 96  # > NUM_LAYERS so contiguous arange indices take the merged path
    PAGE_SIZE = 128
    HEADS = 1
    HEAD_DIM = 64

    def _value(self, layer, page):
        # Distinct per (layer, page); float32 keeps it exact.
        return float((layer + 1) * 1000 + page)

    def _make_buffers(self):
        device_k = torch.zeros(
            (self.NUM_LAYERS, self.NUM_PAGES, self.PAGE_SIZE, self.HEADS, self.HEAD_DIM),
            dtype=torch.float32,
            device="npu",
        )
        host_k = torch.zeros(
            (self.NUM_PAGES, self.NUM_LAYERS, self.PAGE_SIZE, self.HEADS, self.HEAD_DIM),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        return device_k, host_k

    def _fill_device(self, device_k):
        for layer in range(self.NUM_LAYERS):
            for page in range(self.NUM_PAGES):
                device_k[layer, page].fill_(self._value(layer, page))

    def _fill_host(self, host_k):
        for layer in range(self.NUM_LAYERS):
            for page in range(self.NUM_PAGES):
                host_k[page, layer].fill_(self._value(layer, page))

    def _check_host(self, host_k, covered_pages):
        host_cpu = host_k.cpu()
        for page in range(self.NUM_PAGES):
            for layer in range(self.NUM_LAYERS):
                expected = self._value(layer, page) if page in covered_pages else 0.0
                if not torch.all(torch.eq(host_cpu[page, layer], expected)):
                    self.fail(f"host[{page}, {layer}] mismatch after D2H run transfer")

    def _check_device(self, device_k, covered_pages):
        device_cpu = device_k.cpu()
        for layer in range(self.NUM_LAYERS):
            for page in range(self.NUM_PAGES):
                expected = self._value(layer, page) if page in covered_pages else 0.0
                if not torch.all(torch.eq(device_cpu[layer, page], expected)):
                    self.fail(f"device[{layer}, {page}] mismatch after H2D run transfer")

    def test_contiguous_run_merged_d2h(self):
        torch.npu.set_device(0)
        device_k, host_k = self._make_buffers()
        self._fill_device(device_k)
        indices = torch.arange(self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64)
        with torch.npu.stream(torch.npu.Stream()):
            transfer_kv_dim_exchange(
                device_indices=indices,
                host_indices=indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.D2H,
            )
        torch.npu.synchronize()
        self._check_host(host_k, set(range(self.NUM_PAGES)))

    def test_contiguous_run_merged_h2d(self):
        torch.npu.set_device(0)
        device_k, host_k = self._make_buffers()
        self._fill_host(host_k)
        indices = torch.arange(self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64)
        with torch.npu.stream(torch.npu.Stream()):
            transfer_kv_dim_exchange(
                device_indices=indices,
                host_indices=indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
            )
        torch.npu.synchronize()
        self._check_device(device_k, set(range(self.NUM_PAGES)))

    def _segment_indices(self, segments):
        """Build token-level indices from (host_page0, device_page0, n) segments."""
        device_idx, host_idx = [], []
        for host_p0, dev_p0, n in segments:
            host_idx.extend(range(host_p0 * self.PAGE_SIZE, (host_p0 + n) * self.PAGE_SIZE))
            device_idx.extend(range(dev_p0 * self.PAGE_SIZE, (dev_p0 + n) * self.PAGE_SIZE))
        return (
            torch.tensor(device_idx, dtype=torch.int64),
            torch.tensor(host_idx, dtype=torch.int64),
        )

    def test_mixed_segments_d2h(self):
        # Two long runs (32 and 24 pages >= 16 layers -> merged form) and two
        # short runs (2 pages < 16 -> per-page fallback), with host/device page
        # offsets deliberately different inside each run.
        torch.npu.set_device(0)
        segments = [
            (0, 64, 32),   # merged
            (40, 0, 2),    # fallback
            (60, 8, 24),   # merged
            (90, 2, 2),    # fallback
        ]
        covered_host_pages = set()
        for host_p0, _, n in segments:
            covered_host_pages.update(range(host_p0, host_p0 + n))

        device_k, host_k = self._make_buffers()
        self._fill_device(device_k)
        device_indices, host_indices = self._segment_indices(segments)
        with torch.npu.stream(torch.npu.Stream()):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.D2H,
            )
        torch.npu.synchronize()
        self._check_host(host_k, covered_host_pages)

    def test_mixed_segments_h2d(self):
        torch.npu.set_device(0)
        segments = [
            (0, 64, 32),
            (40, 0, 2),
            (60, 8, 24),
            (90, 2, 2),
        ]
        covered_device_pages = set()
        for _, dev_p0, n in segments:
            covered_device_pages.update(range(dev_p0, dev_p0 + n))

        device_k, host_k = self._make_buffers()
        self._fill_host(host_k)
        device_indices, host_indices = self._segment_indices(segments)
        with torch.npu.stream(torch.npu.Stream()):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
            )
        torch.npu.synchronize()
        self._check_device(device_k, covered_device_pages)

    def test_run_exceeding_pages_rejected(self):
        torch.npu.set_device(0)
        device_k, host_k = self._make_buffers()
        # A run that runs past the end of the device page space must fail the
        # bounds check: last 8 device pages starting at 90 would end at 97 > 96.
        segments = [(0, 90, 8)]
        device_indices, host_indices = self._segment_indices(segments)
        with self.assertRaises(RuntimeError):
            transfer_kv_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=torch.empty(0),
                host_v=torch.empty(0),
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
            )


class TestTransferKVAscendCTable(unittest.TestCase):
    """acc_offload AIV sparse-copy path (transfer_kv_dim_exchange_table +
    offload.sparse_copy) vs the legacy memcpy2d path.

    The AIV kernel de-references host pointers directly, so its host pool is
    hybm-backed (offload.empty); the reference path uses ordinary pinned
    memory.  Both an A/B comparison against the memcpy2d result and an
    absolute per-(layer, page) value check are performed, with a shuffled
    host page mapping so wrong page/layer pitches cannot cancel out.
    """

    NUM_LAYERS = 8
    INDEX_LAYERS = 3  # DSA-style: fewer indexer layers than k/v layers
    NUM_PAGES = 16
    PAGE_SIZE = 128
    K_WIDTH = 64
    V_WIDTH = 32
    INDEX_WIDTH = 128
    ONE_GB = 1 << 30

    def _value(self, layer, page):
        # Distinct and exact in bf16 (integers < 256).
        return float(layer * 17 + page + 1)

    @classmethod
    def _reserve_bytes(cls):
        ps = cls.PAGE_SIZE
        return (
            cls.NUM_PAGES
            * ps
            * (
                cls.NUM_LAYERS * (cls.K_WIDTH + cls.V_WIDTH)
                + cls.INDEX_LAYERS * cls.INDEX_WIDTH
            )
            * 2
            + cls.NUM_PAGES * ps * cls.INDEX_LAYERS * 4  # FP32 scale
        )

    @classmethod
    def setUpClass(cls):
        if _mf_offload is None:
            raise unittest.SkipTest("memfabric_hybrid is not installed")
        if transfer_kv_dim_exchange_table is None:
            raise unittest.SkipTest(
                "transfer_kv_dim_exchange_table is unavailable; rebuild sgl-kernel-npu"
            )
        torch.npu.set_device(0)
        cls.offload = _mf_offload
        reserve = max(cls.ONE_GB, ((cls._reserve_bytes() + cls.ONE_GB - 1) // cls.ONE_GB) * cls.ONE_GB)
        config = _mf_offload.OffloadConfig()
        config.device_id = 0
        config.reserve_size = reserve
        config.alloc_size = reserve
        assert _mf_offload.initialize(config) == 0, "offload.initialize failed"
        cls.buffers = None  # allocated per test to control lifetimes

    @classmethod
    def tearDownClass(cls):
        if _mf_offload is not None and cls.offload is _mf_offload:
            cls.buffers = None
            _mf_offload.uninitialize()

    # -- buffers ----------------------------------------------------------
    def _alloc(self):
        ps, L, IL, P = self.PAGE_SIZE, self.NUM_LAYERS, self.INDEX_LAYERS, self.NUM_PAGES
        dev = lambda layers, width, dtype=torch.bfloat16: torch.zeros(
            (layers, P, ps, 1, width), dtype=dtype, device="npu"
        )
        hyb = lambda layers, width, dtype=torch.bfloat16: self.offload.empty(
            [P, layers, ps, 1, width], dtype=dtype
        ).zero_()
        pin = lambda layers, width, dtype=torch.bfloat16: torch.zeros(
            (P, layers, ps, 1, width), dtype=dtype, device="cpu", pin_memory=True
        )
        self.buffers = {
            "device_k": dev(L, self.K_WIDTH),
            "device_v": dev(L, self.V_WIDTH),
            "device_index_k": dev(IL, self.INDEX_WIDTH),
            "device_scale": torch.zeros((IL, P, ps, 1), dtype=torch.float32, device="npu"),
            "host_k": hyb(L, self.K_WIDTH),
            "host_v": hyb(L, self.V_WIDTH),
            "host_index_k": hyb(IL, self.INDEX_WIDTH),
            "host_scale": hyb(IL, 1, torch.float32),
            "ref_k": pin(L, self.K_WIDTH),
            "ref_v": pin(L, self.V_WIDTH),
            "ref_index_k": pin(IL, self.INDEX_WIDTH),
            "ref_scale": pin(IL, 1, torch.float32),
        }

    def _indices(self):
        # device page i maps to host page perm[i]; token-level expansion.
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(self.NUM_PAGES, generator=g)
        host_tokens = (
            (perm * self.PAGE_SIZE).repeat_interleave(self.PAGE_SIZE)
            + torch.arange(self.PAGE_SIZE).repeat(self.NUM_PAGES)
        )
        device_indices = torch.arange(
            self.NUM_PAGES * self.PAGE_SIZE, dtype=torch.int64
        )
        return device_indices, host_tokens.to(device_indices.device), perm

    # -- transfer helpers --------------------------------------------------
    def _run_memcpy2d(self, bufs, direction, host_key_prefix="ref"):
        transfer_kv_dim_exchange(
            device_indices=bufs["device_indices"],
            host_indices=bufs["host_indices"],
            device_k=bufs["device_k"],
            host_k=bufs[f"{host_key_prefix}_k"],
            device_v=bufs["device_v"],
            host_v=bufs[f"{host_key_prefix}_v"],
            device_index_k=bufs["device_index_k"],
            host_index_k=bufs[f"{host_key_prefix}_index_k"],
            device_index_k_scale=bufs["device_scale"],
            host_index_k_scale=bufs[f"{host_key_prefix}_scale"],
            page_size=self.PAGE_SIZE,
            direction=direction,
        )

    def _run_aiv(self, bufs, direction):
        src, dst, lens, size = transfer_kv_dim_exchange_table(
            device_indices=bufs["device_indices"],
            host_indices=bufs["host_indices"],
            device_k=bufs["device_k"],
            host_k=bufs["host_k"],
            device_v=bufs["device_v"],
            host_v=bufs["host_v"],
            device_index_k=bufs["device_index_k"],
            host_index_k=bufs["host_index_k"],
            device_index_k_scale=bufs["device_scale"],
            host_index_k_scale=bufs["host_scale"],
            page_size=self.PAGE_SIZE,
            direction=direction,
        )
        device = torch.device("npu", torch.npu.current_device())
        ret = self.offload.sparse_copy(src, dst, lens, size, device)
        self.assertEqual(ret, 0, "offload.sparse_copy failed")
        stream = torch.npu.current_stream()
        for t in (src, dst, lens, size):
            t.record_stream(stream)

    # -- checks ------------------------------------------------------------
    def _fill_device(self, bufs):
        for key, layers in (
            ("device_k", self.NUM_LAYERS),
            ("device_v", self.NUM_LAYERS),
            ("device_index_k", self.INDEX_LAYERS),
            ("device_scale", self.INDEX_LAYERS),
        ):
            for layer in range(layers):
                for page in range(self.NUM_PAGES):
                    bufs[key][layer, page].fill_(self._value(layer, page))

    def _fill_host(self, bufs):
        perm = bufs["perm"].tolist()
        for key, layers in (
            ("host_k", self.NUM_LAYERS),
            ("host_v", self.NUM_LAYERS),
            ("host_index_k", self.INDEX_LAYERS),
            ("host_scale", self.INDEX_LAYERS),
        ):
            for layer in range(layers):
                for dp, hp in enumerate(perm):
                    bufs[key][hp, layer].fill_(self._value(layer, dp))

    def _check_host_absolute(self, host, layers, label):
        # host[perm[dp], layer] must equal value(layer, dp) everywhere.
        perm = self.buffers["perm"].tolist()
        for layer in range(layers):
            for dp in range(self.NUM_PAGES):
                expect = self._value(layer, dp)
                got = host[perm[dp], layer]
                self.assertTrue(
                    torch.all(torch.eq(got, expect)),
                    f"{label} mismatch at layer={layer} page={dp}",
                )

    def _check_device_absolute(self, device, layers, label):
        cpu = device.cpu()
        for layer in range(layers):
            for page in range(self.NUM_PAGES):
                expect = self._value(layer, page)
                self.assertTrue(
                    torch.all(torch.eq(cpu[layer, page], expect)),
                    f"{label} mismatch at layer={layer} page={page}",
                )

    # -- tests --------------------------------------------------------------
    def test_d2h_aiv_matches_memcpy2d(self):
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        self._fill_device(self.buffers)
        self._run_memcpy2d(self.buffers, TransferDirection.D2H)
        torch.npu.synchronize()
        # Absolute check on the reference path, then wipe and run AIV.
        for key, layers in (
            ("ref_k", self.NUM_LAYERS),
            ("ref_v", self.NUM_LAYERS),
            ("ref_index_k", self.INDEX_LAYERS),
            ("ref_scale", self.INDEX_LAYERS),
        ):
            self._check_host_absolute(self.buffers[key], layers, key)
        for key in ("host_k", "host_v", "host_index_k", "host_scale"):
            self.buffers[key].zero_()

        self._run_aiv(self.buffers, TransferDirection.D2H)
        torch.npu.synchronize()
        for hyb, ref in (
            ("host_k", "ref_k"),
            ("host_v", "ref_v"),
            ("host_index_k", "ref_index_k"),
            ("host_scale", "ref_scale"),
        ):
            self.assertTrue(
                torch.equal(self.buffers[hyb], self.buffers[ref]),
                f"AIV D2H result differs from memcpy2d for {hyb}",
            )

    def test_h2d_aiv_matches_memcpy2d(self):
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        # Fill the hybm host pool; reference: memcpy2d into wiped device
        # buffers (fresh copies), then wipe again and run AIV.
        self._fill_host(self.buffers)
        # Both paths read the same hybm host pool as the H2D source.
        self._run_memcpy2d(self.buffers, TransferDirection.H2D, host_key_prefix="host")
        torch.npu.synchronize()
        # Absolute check on the reference path, then wipe and run AIV.
        for key, layers in (
            ("device_k", self.NUM_LAYERS),
            ("device_v", self.NUM_LAYERS),
            ("device_index_k", self.INDEX_LAYERS),
            ("device_scale", self.INDEX_LAYERS),
        ):
            self._check_device_absolute(self.buffers[key], layers, key)
        ref = {
            k: self.buffers[k].detach().cpu().clone()
            for k in ("device_k", "device_v", "device_index_k", "device_scale")
        }
        for k in ("device_k", "device_v", "device_index_k", "device_scale"):
            self.buffers[k].zero_()

        self._run_aiv(self.buffers, TransferDirection.H2D)
        torch.npu.synchronize()
        for k, ref_t in ref.items():
            self.assertTrue(
                torch.equal(self.buffers[k].cpu(), ref_t),
                f"AIV H2D result differs from memcpy2d for {k}",
            )

    def test_table_entry_count_and_split(self):
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        src, dst, lens, size = transfer_kv_dim_exchange_table(
            device_indices=device_indices,
            host_indices=host_indices,
            device_k=self.buffers["device_k"],
            host_k=self.buffers["host_k"],
            device_v=self.buffers["device_v"],
            host_v=self.buffers["host_v"],
            device_index_k=self.buffers["device_index_k"],
            host_index_k=self.buffers["host_index_k"],
            device_index_k_scale=self.buffers["device_scale"],
            host_index_k_scale=self.buffers["host_scale"],
            page_size=self.PAGE_SIZE,
            direction=TransferDirection.D2H,
        )
        P, L, IL = self.NUM_PAGES, self.NUM_LAYERS, self.INDEX_LAYERS
        # No row exceeds 88KB with these widths, so no entry is split.
        expected = P * L + P * L + P * IL + P * IL  # k + v + index_k + scale
        self.assertEqual(int(size.cpu().item()), expected)
        self.assertEqual(src.numel(), expected)
        self.assertEqual(dst.numel(), expected)
        self.assertEqual(lens.numel(), expected)
        # Row widths per component (bytes): k, v, index_k, scale.
        w = lambda width, itemsize=2: self.PAGE_SIZE * width * itemsize
        lens_cpu = lens.cpu()
        for width, count in (
            (w(self.K_WIDTH), P * L),
            (w(self.V_WIDTH), P * L),
            (w(self.INDEX_WIDTH), P * IL),
            (w(1, 4), P * IL),
        ):
            self.assertEqual(int((lens_cpu == width).sum()), count, f"len entries for width={width}")

    def test_layer_group_pipeline_d2h(self):
        """Per-group table builds + sparse_copy reproduce the one-shot result.

        k/v layers split into groups [0,3), [3,6), [6,8) with independently
        chosen index_k/scale slot ranges [0,1), [1,3), (skipped) -- together
        they must cover every k/v and index_k/scale layer exactly once and
        match the one-shot AIV transfer element-wise.
        """
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        self._fill_device(self.buffers)

        # Reference: one-shot AIV D2H.
        self._run_aiv(self.buffers, TransferDirection.D2H)
        torch.npu.synchronize()
        ref = {
            k: self.buffers[k].clone()
            for k in ("host_k", "host_v", "host_index_k", "host_scale")
        }
        for k in ref:
            self.buffers[k].zero_()

        # Pipelined: per-group builds + launches (all on one stream, exactly
        # how the scheduler's layer loop interleaves them).
        groups = [
            # (layer_start, layer_num, index_k_layer_start, index_k_layer_num)
            (0, 3, 0, 1),
            (3, 3, 1, 2),
            (6, 2, 0, 0),  # trailing group without indexer layers
        ]
        for layer_start, layer_num, ik_start, ik_num in groups:
            src, dst, lens, size = transfer_kv_dim_exchange_table(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=self.buffers["device_k"],
                host_k=self.buffers["host_k"],
                device_v=self.buffers["device_v"],
                host_v=self.buffers["host_v"],
                device_index_k=self.buffers["device_index_k"],
                host_index_k=self.buffers["host_index_k"],
                device_index_k_scale=self.buffers["device_scale"],
                host_index_k_scale=self.buffers["host_scale"],
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.D2H,
                layer_start=layer_start,
                layer_num=layer_num,
                index_k_layer_start=ik_start,
                index_k_layer_num=ik_num,
            )
            ret = self.offload.sparse_copy(
                src, dst, lens, size, torch.device("npu", torch.npu.current_device())
            )
            self.assertEqual(ret, 0, "offload.sparse_copy failed")
            stream = torch.npu.current_stream()
            for t in (src, dst, lens, size):
                t.record_stream(stream)
        torch.npu.synchronize()

        for k, ref_t in ref.items():
            self.assertTrue(
                torch.equal(self.buffers[k], ref_t),
                f"layer-group D2H result differs from one-shot for {k}",
            )

    def test_layer_group_pipeline_h2d(self):
        """Same as test_layer_group_pipeline_d2h for the H2D direction."""
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        self._fill_host(self.buffers)

        self._run_aiv(self.buffers, TransferDirection.H2D)
        torch.npu.synchronize()
        ref = {
            k: self.buffers[k].detach().cpu().clone()
            for k in ("device_k", "device_v", "device_index_k", "device_scale")
        }
        for k in ref:
            self.buffers[k].zero_()

        groups = [(0, 3, 0, 1), (3, 3, 1, 2), (6, 2, 0, 0)]
        for layer_start, layer_num, ik_start, ik_num in groups:
            src, dst, lens, size = transfer_kv_dim_exchange_table(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=self.buffers["device_k"],
                host_k=self.buffers["host_k"],
                device_v=self.buffers["device_v"],
                host_v=self.buffers["host_v"],
                device_index_k=self.buffers["device_index_k"],
                host_index_k=self.buffers["host_index_k"],
                device_index_k_scale=self.buffers["device_scale"],
                host_index_k_scale=self.buffers["host_scale"],
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.H2D,
                layer_start=layer_start,
                layer_num=layer_num,
                index_k_layer_start=ik_start,
                index_k_layer_num=ik_num,
            )
            ret = self.offload.sparse_copy(
                src, dst, lens, size, torch.device("npu", torch.npu.current_device())
            )
            self.assertEqual(ret, 0, "offload.sparse_copy failed")
            stream = torch.npu.current_stream()
            for t in (src, dst, lens, size):
                t.record_stream(stream)
        torch.npu.synchronize()

        for k, ref_t in ref.items():
            self.assertTrue(
                torch.equal(self.buffers[k].cpu(), ref_t),
                f"layer-group H2D result differs from one-shot for {k}",
            )

    def test_layer_range_entry_count(self):
        """Layer ranges shrink the table by exactly the excluded layers."""
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        src, dst, lens, size = transfer_kv_dim_exchange_table(
            device_indices=device_indices,
            host_indices=host_indices,
            device_k=self.buffers["device_k"],
            host_k=self.buffers["host_k"],
            device_v=self.buffers["device_v"],
            host_v=self.buffers["host_v"],
            device_index_k=self.buffers["device_index_k"],
            host_index_k=self.buffers["host_index_k"],
            device_index_k_scale=self.buffers["device_scale"],
            host_index_k_scale=self.buffers["host_scale"],
            page_size=self.PAGE_SIZE,
            direction=TransferDirection.D2H,
            layer_start=2,
            layer_num=3,
            index_k_layer_start=1,
            index_k_layer_num=2,
        )
        P = self.NUM_PAGES
        # k + v: 3 layers; index_k + scale: 2 layers; no row exceeds 88KB.
        expected = P * 3 + P * 3 + P * 2 + P * 2
        self.assertEqual(int(size.cpu().item()), expected)
        self.assertEqual(lens.numel(), expected)

    def test_layer_range_out_of_bounds_rejected(self):
        self._alloc()
        device_indices, host_indices, perm = self._indices()
        self.buffers.update(device_indices=device_indices, host_indices=host_indices, perm=perm)
        with self.assertRaises(RuntimeError):
            transfer_kv_dim_exchange_table(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=self.buffers["device_k"],
                host_k=self.buffers["host_k"],
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.D2H,
                layer_start=6,   # 6 + 4 > 8 layers
                layer_num=4,
            )
        with self.assertRaises(RuntimeError):
            transfer_kv_dim_exchange_table(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=self.buffers["device_k"],
                host_k=self.buffers["host_k"],
                device_index_k=self.buffers["device_index_k"],
                host_index_k=self.buffers["host_index_k"],
                page_size=self.PAGE_SIZE,
                direction=TransferDirection.D2H,
                index_k_layer_start=2,  # 2 + 2 > 3 indexer layers
                index_k_layer_num=2,
            )

    def test_wide_row_entry_split(self):
        """Rows wider than the 88KB UB buffer must split into two entries."""
        P, L, PS, WIDE = 20, 2, 128, 512  # row = 128*512*2B = 128KB -> 2 entries
        device_k = torch.zeros((L, P, PS, 1, WIDE), dtype=torch.bfloat16, device="npu")
        host_k = self.offload.empty([P, L, PS, 1, WIDE], dtype=torch.bfloat16).zero_()
        ref_k = torch.zeros((P, L, PS, 1, WIDE), dtype=torch.bfloat16, device="cpu", pin_memory=True)

        for layer in range(L):
            for page in range(P):
                device_k[layer, page].fill_(self._value(layer, page))

        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(P, generator=g)
        host_tokens = (perm * PS).repeat_interleave(PS) + torch.arange(PS).repeat(P)
        device_indices = torch.arange(P * PS, dtype=torch.int64)
        host_indices = host_tokens.to(device_indices.device)

        transfer_kv_dim_exchange(
            device_indices=device_indices,
            host_indices=host_indices,
            device_k=device_k,
            host_k=ref_k,
            device_v=torch.empty(0),
            host_v=torch.empty(0),
            page_size=PS,
            direction=TransferDirection.D2H,
        )
        src, dst, lens, size = transfer_kv_dim_exchange_table(
            device_indices=device_indices,
            host_indices=host_indices,
            device_k=device_k,
            host_k=host_k,
            page_size=PS,
            direction=TransferDirection.D2H,
        )
        self.assertEqual(int(size.cpu().item()), P * L * 2, "each 128KB row must split into 2 entries")
        ret = self.offload.sparse_copy(
            src, dst, lens, size, torch.device("npu", torch.npu.current_device())
        )
        self.assertEqual(ret, 0)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(host_k, ref_k), "split-entry AIV copy differs from memcpy2d")
        perm_list = perm.tolist()
        for layer in range(L):
            for dp in range(P):
                self.assertTrue(
                    torch.all(torch.eq(host_k[perm_list[dp], layer], self._value(layer, dp))),
                    f"wide-row mismatch at layer={layer} page={dp}",
                )


if __name__ == "__main__":
    unittest.main()
