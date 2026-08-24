import time
import unittest

import torch
from sgl_kernel_npu.kvcacheio import (
    TransferDirection,
    TransferFlag,
    transfer_kv_dim_exchange,
)

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


if __name__ == "__main__":
    unittest.main()
