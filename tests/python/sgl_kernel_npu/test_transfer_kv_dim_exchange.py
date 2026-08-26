import time
import unittest

import torch
from sgl_kernel_npu.kvcacheio import (
    TransferDirection,
    TransferFlag,
    transfer_kv_dim_exchange,
    transfer_kv_per_layer_dim_exchange,
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


class TestTransferKVPerLayer(unittest.TestCase):
    """Per-layer dim_exchange transfer with contiguous-run merging.

    Uses small buffers: layer-first device layout (L, P, page_size, H, D) and
    page-first host layout (P, L, page_size, H, D).  Every (layer, page) slot
    holds a unique exact fp32 integer so that a wrong layer/page offset or
    pitch -- the layout-sensitive part of this operator -- fails the
    element-wise assertions below.
    """

    L = 8
    P = 16
    H = 4
    D = 128
    LAYER_ID = 7  # last layer: also exercises the layer-count boundary

    # Indexer components live in their own slot space: slot 3 maps to k/v
    # layer 7 in this test (mirrors partial-indexer models like GLM 5.2).
    INDEXER_SLOT = 3
    INDEXER_LAYERS = 5

    # Three runs: (3,4,5)<->(7,8,9) merges into one aclrtMemcpy2dAsync,
    # (10,11)<->(2,3) into another, and (0)<->(5) stays a singleton.
    DEVICE_PAGES = [3, 4, 5, 10, 11, 0]
    HOST_PAGES = [7, 8, 9, 2, 3, 5]

    def _values(self, layers, pages, base):
        # value = base + (layer * pages + page); unique and exact in fp32.
        vals = torch.arange(layers * pages, dtype=torch.float32).view(
            layers, pages, 1, 1, 1
        ).add_(base)
        return vals.expand(layers, pages, PAGE_SIZE, self.H, self.D).contiguous()

    def _token_indices(self, pages):
        page_ids = torch.tensor(pages, dtype=torch.int64)
        return (
            page_ids.repeat_interleave(PAGE_SIZE) * PAGE_SIZE
            + torch.arange(PAGE_SIZE, dtype=torch.int64).repeat(len(pages))
        )

    def _make_kv(self):
        device_k = self._values(self.L, self.P, base=1.0).to("npu")
        device_v = self._values(self.L, self.P, base=3.0).to("npu")
        host_k = self._values(self.L, self.P, base=2.0).pin_memory()
        host_v = self._values(self.L, self.P, base=4.0).pin_memory()
        return device_k, host_k, device_v, host_v

    def _expected_d2h(self, host, device, layer, device_pages, host_pages):
        """host after D2H: moved slots hold the device value, rest untouched."""
        expected = host.clone()
        dev = device.cpu()
        for d, h in zip(device_pages, host_pages):
            expected[h, layer] = dev[layer, d]
        return expected

    def _expected_h2d(self, device, host, layer, device_pages, host_pages):
        """device after H2D: moved slots hold the host value, rest untouched."""
        expected = device.cpu().clone()
        for d, h in zip(device_pages, host_pages):
            expected[layer, d] = host[h, layer]
        return expected

    def _per_layer_transfer(
        self,
        direct,
        device_pages,
        host_pages,
        layer_id=LAYER_ID,
        device_index_k=None,
        host_index_k=None,
        device_index_k_scale=None,
        host_index_k_scale=None,
        index_k_layer_id=None,
    ):
        torch.npu.set_device(0)
        device_k, host_k, device_v, host_v = self._make_kv()
        device_indices = self._token_indices(device_pages)
        host_indices = self._token_indices(host_pages)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            transfer_kv_per_layer_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=device_v,
                host_v=host_v,
                layer_id=layer_id,
                device_index_k=device_index_k,
                host_index_k=host_index_k,
                device_index_k_scale=device_index_k_scale,
                host_index_k_scale=host_index_k_scale,
                index_k_layer_id=index_k_layer_id,
                page_size=PAGE_SIZE,
                direction=direct,
            )
        torch.npu.synchronize()
        return device_k, host_k, device_v, host_v

    def test_per_layer_kv_copy_d2h(self):
        identity = list(range(self.P))
        device_k, host_k, _, _ = self._per_layer_transfer(
            TransferDirection.D2H, identity, identity
        )
        self.assertTrue(
            torch.equal(host_k, self._expected_d2h(host_k, device_k, self.LAYER_ID, identity, identity)),
            msg="host k should hold device k values for the moved layer only",
        )

    def test_per_layer_kv_copy_h2d(self):
        identity = list(range(self.P))
        device_k, host_k, _, _ = self._per_layer_transfer(
            TransferDirection.H2D, identity, identity
        )
        self.assertTrue(
            torch.equal(device_k.cpu(), self._expected_h2d(device_k, host_k, self.LAYER_ID, identity, identity)),
            msg="device k should hold host k values for the moved layer only",
        )

    def test_per_layer_run_merge_d2h(self):
        device_k, host_k, _, _ = self._per_layer_transfer(
            TransferDirection.D2H, self.DEVICE_PAGES, self.HOST_PAGES
        )
        self.assertTrue(
            torch.equal(host_k, self._expected_d2h(host_k, device_k, self.LAYER_ID, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="host k should hold device k values at the mapped pages for the moved layer",
        )

    def test_per_layer_run_merge_h2d(self):
        device_k, host_k, _, _ = self._per_layer_transfer(
            TransferDirection.H2D, self.DEVICE_PAGES, self.HOST_PAGES
        )
        self.assertTrue(
            torch.equal(device_k.cpu(), self._expected_h2d(device_k, host_k, self.LAYER_ID, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="device k should hold host k values at the mapped pages for the moved layer",
        )

    def test_per_layer_indexer_slot_d2h(self):
        # k/v move layer LAYER_ID while index_k moves INDEXER_SLOT: the slot
        # id differs from the layer id, exercising the separate layer space.
        device_index_k = self._values(self.INDEXER_LAYERS, self.P, base=5.0).to("npu")
        host_index_k = self._values(self.INDEXER_LAYERS, self.P, base=6.0).pin_memory()
        device_k, host_k, _, _ = self._per_layer_transfer(
            TransferDirection.D2H,
            self.DEVICE_PAGES,
            self.HOST_PAGES,
            layer_id=self.LAYER_ID,
            device_index_k=device_index_k,
            host_index_k=host_index_k,
            index_k_layer_id=self.INDEXER_SLOT,
        )
        self.assertTrue(
            torch.equal(host_k, self._expected_d2h(host_k, device_k, self.LAYER_ID, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="host k should hold device k values for the moved layer",
        )
        self.assertTrue(
            torch.equal(
                host_index_k,
                self._expected_d2h(host_index_k, device_index_k, self.INDEXER_SLOT, self.DEVICE_PAGES, self.HOST_PAGES),
            ),
            msg="host index_k should hold device index_k values at the indexer slot for the moved pages",
        )

    def test_per_layer_indexer_skip(self):
        # index_k_layer_id=None must skip the indexer components entirely.
        device_index_k = self._values(self.INDEXER_LAYERS, self.P, base=5.0).to("npu")
        host_index_k = self._values(self.INDEXER_LAYERS, self.P, base=6.0).pin_memory()
        _, _, _, _ = self._per_layer_transfer(
            TransferDirection.D2H,
            self.DEVICE_PAGES,
            self.HOST_PAGES,
            layer_id=self.LAYER_ID,
            device_index_k=device_index_k,
            host_index_k=host_index_k,
            index_k_layer_id=None,
        )
        self.assertTrue(
            torch.equal(host_index_k, self._values(self.INDEXER_LAYERS, self.P, base=6.0)),
            msg="host index_k must stay untouched when index_k_layer_id is None",
        )

    def test_per_layer_scale_slot_d2h(self):
        # Device scale is 4-D (layers, pages, page_size, 1); the wrapper pads
        # the trailing singleton dim.  Host scale is 5-D page-first.
        device_scale = (
            torch.arange(self.INDEXER_LAYERS * self.P, dtype=torch.float32)
            .view(self.INDEXER_LAYERS, self.P, 1, 1)
            .add_(7.0)
            .expand(self.INDEXER_LAYERS, self.P, PAGE_SIZE, 1)
            .contiguous()
            .to("npu")
        )
        host_scale = (
            torch.arange(self.INDEXER_LAYERS * self.P, dtype=torch.float32)
            .view(self.INDEXER_LAYERS, self.P)
            .add_(8.0)
            .permute(1, 0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(self.P, self.INDEXER_LAYERS, PAGE_SIZE, 1, 1)
            .contiguous()
            .pin_memory()
        )
        torch.npu.set_device(0)
        device_k, host_k, device_v, host_v = self._make_kv()
        device_indices = self._token_indices(self.DEVICE_PAGES)
        host_indices = self._token_indices(self.HOST_PAGES)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            transfer_kv_per_layer_dim_exchange(
                device_indices=device_indices,
                host_indices=host_indices,
                device_k=device_k,
                host_k=host_k,
                device_v=device_v,
                host_v=host_v,
                layer_id=self.LAYER_ID,
                device_index_k_scale=device_scale,
                host_index_k_scale=host_scale,
                index_k_layer_id=self.INDEXER_SLOT,
                page_size=PAGE_SIZE,
                direction=TransferDirection.D2H,
            )
        torch.npu.synchronize()

        expected = host_scale.clone()
        dev = device_scale.cpu()
        for d, h in zip(self.DEVICE_PAGES, self.HOST_PAGES):
            expected[h, self.INDEXER_SLOT] = dev[self.INDEXER_SLOT, d].unsqueeze(-1)
        self.assertTrue(
            torch.equal(host_scale, expected),
            msg="host scale should hold device scale values at the indexer slot for the moved pages",
        )

    def test_per_layer_layer_out_of_range(self):
        for bad_layer in (self.L, -1):
            with self.assertRaises(RuntimeError):
                self._per_layer_transfer(
                    TransferDirection.D2H,
                    self.DEVICE_PAGES,
                    self.HOST_PAGES,
                    layer_id=bad_layer,
                )


class TestTransferKVAllLayerAdaptive(unittest.TestCase):
    """All-layer dim_exchange with the adaptive min(run, layer_num) strategy.

    Small buffers with L=4 layers so that contiguous runs of >= 4 pages take
    the merged form (one copy per layer, rows are pages) while shorter runs
    keep the per-page form (one copy per page, rows are layers).  Every
    (layer, page) slot holds a unique exact fp32 value so a wrong layer/page
    offset or pitch fails the element-wise assertions.
    """

    L = 4
    P = 16
    H = 2
    D = 128

    # Runs: (3..8)<->(9..14) has length 6 >= L (merged form), (10,11)<->(0,1)
    # has length 2 < L (per-page form) and (14)<->(5) is a singleton; the
    # page mapping is scattered so a wrong page offset cannot cancel out.
    DEVICE_PAGES = [3, 4, 5, 6, 7, 8, 10, 11, 14]
    HOST_PAGES = [9, 10, 11, 12, 13, 14, 0, 1, 5]

    def _values(self, layers, pages, base):
        vals = torch.arange(layers * pages, dtype=torch.float32).view(layers, pages, 1, 1, 1).add_(base)
        return vals.expand(layers, pages, PAGE_SIZE, self.H, self.D).contiguous()

    def _token_indices(self, pages):
        page_ids = torch.tensor(pages, dtype=torch.int64)
        return (
            page_ids.repeat_interleave(PAGE_SIZE) * PAGE_SIZE
            + torch.arange(PAGE_SIZE, dtype=torch.int64).repeat(len(pages))
        )

    def _make_kv(self):
        device_k = self._values(self.L, self.P, base=1.0).to("npu")
        device_v = self._values(self.L, self.P, base=3.0).to("npu")
        host_k = self._values(self.L, self.P, base=2.0).pin_memory()
        host_v = self._values(self.L, self.P, base=4.0).pin_memory()
        return device_k, host_k, device_v, host_v

    def _expected_d2h(self, host, device, device_pages, host_pages):
        """host after D2H: moved pages hold the device values, rest untouched."""
        expected = host.clone()
        dev = device.cpu()
        for d, h in zip(device_pages, host_pages):
            expected[h] = dev[:, d]
        return expected

    def _expected_h2d(self, device, host, device_pages, host_pages):
        """device after H2D: moved pages hold the host values, rest untouched."""
        expected = device.cpu().clone()
        for d, h in zip(device_pages, host_pages):
            expected[:, d] = host[h]
        return expected

    def _expected_d2h_range(self, host, device, device_pages, host_pages, layer_start, layer_num):
        """host after a ranged D2H: only [layer_start, layer_start+layer_num) moves."""
        expected = host.clone()
        dev = device.cpu()
        for d, h in zip(device_pages, host_pages):
            expected[h, layer_start : layer_start + layer_num] = dev[layer_start : layer_start + layer_num, d]
        return expected

    def _expected_h2d_range(self, device, host, device_pages, host_pages, layer_start, layer_num):
        """device after a ranged H2D: only [layer_start, layer_start+layer_num) moves."""
        expected = device.cpu().clone()
        for d, h in zip(device_pages, host_pages):
            expected[layer_start : layer_start + layer_num, d] = host[h, layer_start : layer_start + layer_num]
        return expected

    def _all_layer_transfer(
        self,
        direct,
        device_pages=None,
        host_pages=None,
        with_index_k=False,
        layer_start=0,
        layer_num=-1,
    ):
        torch.npu.set_device(0)
        if device_pages is None:
            device_pages = self.DEVICE_PAGES
        if host_pages is None:
            host_pages = self.HOST_PAGES
        device_k, host_k, device_v, host_v = self._make_kv()
        device_index_k = host_index_k = None
        if with_index_k:
            device_index_k = self._values(self.L, self.P, base=5.0).to("npu")
            host_index_k = self._values(self.L, self.P, base=6.0).pin_memory()
        device_indices = self._token_indices(device_pages)
        host_indices = self._token_indices(host_pages)
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
                layer_start=layer_start,
                layer_num=layer_num,
                page_size=PAGE_SIZE,
                direction=direct,
            )
        torch.npu.synchronize()
        return device_k, host_k, device_v, host_v, device_index_k, host_index_k

    def test_all_layer_mixed_runs_d2h(self):
        # Long run -> merged form, short runs -> per-page form, in one call.
        device_k, host_k, device_v, host_v, _, _ = self._all_layer_transfer(TransferDirection.D2H)
        self.assertTrue(
            torch.equal(host_k, self._expected_d2h(host_k, device_k, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="host k should hold device k values at the mapped pages for every layer",
        )
        self.assertTrue(
            torch.equal(host_v, self._expected_d2h(host_v, device_v, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="host v should hold device v values at the mapped pages for every layer",
        )

    def test_all_layer_mixed_runs_h2d(self):
        device_k, host_k, device_v, host_v, _, _ = self._all_layer_transfer(TransferDirection.H2D)
        self.assertTrue(
            torch.equal(device_k.cpu(), self._expected_h2d(device_k, host_k, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="device k should hold host k values at the mapped pages for every layer",
        )
        self.assertTrue(
            torch.equal(device_v.cpu(), self._expected_h2d(device_v, host_v, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="device v should hold host v values at the mapped pages for every layer",
        )

    def test_all_layer_index_k_mixed_runs_d2h(self):
        # index_k rides the same adaptive path through its own op call and
        # must land element-exact alongside k.
        device_k, host_k, _, _, device_index_k, host_index_k = self._all_layer_transfer(
            TransferDirection.D2H, with_index_k=True
        )
        self.assertTrue(
            torch.equal(host_k, self._expected_d2h(host_k, device_k, self.DEVICE_PAGES, self.HOST_PAGES)),
            msg="host k should hold device k values at the mapped pages for every layer",
        )
        self.assertTrue(
            torch.equal(
                host_index_k,
                self._expected_d2h(host_index_k, device_index_k, self.DEVICE_PAGES, self.HOST_PAGES),
            ),
            msg="host index_k should hold device index_k values at the mapped pages for every layer",
        )

    def test_all_layer_identity_merged_h2d(self):
        # Fully contiguous identity mapping (one run of P pages >= L) drives
        # the pure merged form.
        pages = list(range(self.P))
        device_k, host_k, _, _, _, _ = self._all_layer_transfer(TransferDirection.H2D, pages, pages)
        self.assertTrue(
            torch.equal(device_k.cpu(), self._expected_h2d(device_k, host_k, pages, pages)),
            msg="device k should hold host k values for every layer after the merged-form transfer",
        )

    def test_all_layer_explicit_full_range_h2d(self):
        # Explicit layer_num = L must behave exactly like the default -1.
        pages = list(range(self.P))
        device_k, host_k, _, _, _, _ = self._all_layer_transfer(
            TransferDirection.H2D, pages, pages, layer_start=0, layer_num=self.L
        )
        self.assertTrue(
            torch.equal(device_k.cpu(), self._expected_h2d(device_k, host_k, pages, pages)),
            msg="explicit full layer range should match the all-layer transfer",
        )

    def test_all_layer_subrange_mixed_runs_d2h(self):
        # Middle layer range [1, 3) over mixed runs: layers outside the range
        # and unmapped pages must stay untouched.  With layer_num=2 the
        # length-2 run now equals the height boundary (merged form) and the
        # singleton run falls back to the per-page form.
        layer_start, layer_num = 1, 2
        device_k, host_k, device_v, host_v, _, _ = self._all_layer_transfer(
            TransferDirection.D2H, layer_start=layer_start, layer_num=layer_num
        )
        self.assertTrue(
            torch.equal(
                host_k,
                self._expected_d2h_range(
                    host_k, device_k, self.DEVICE_PAGES, self.HOST_PAGES, layer_start, layer_num
                ),
            ),
            msg="only the requested layer range of host k should hold device values",
        )
        self.assertTrue(
            torch.equal(
                host_v,
                self._expected_d2h_range(
                    host_v, device_v, self.DEVICE_PAGES, self.HOST_PAGES, layer_start, layer_num
                ),
            ),
            msg="only the requested layer range of host v should hold device values",
        )

    def test_all_layer_subrange_tail_h2d(self):
        # Range touching the last layer: [3, 4).
        layer_start, layer_num = 3, 1
        device_k, host_k, _, _, _, _ = self._all_layer_transfer(
            TransferDirection.H2D, layer_start=layer_start, layer_num=layer_num
        )
        self.assertTrue(
            torch.equal(
                device_k.cpu(),
                self._expected_h2d_range(
                    device_k, host_k, self.DEVICE_PAGES, self.HOST_PAGES, layer_start, layer_num
                ),
            ),
            msg="only the last layer of device k should hold host values",
        )

    def test_all_layer_range_validation(self):
        for bad_start, bad_num in ((-1, -1), (0, 0), (2, 3), (self.L, 1), (0, self.L + 1)):
            with self.assertRaises(RuntimeError):
                self._all_layer_transfer(
                    TransferDirection.D2H, layer_start=bad_start, layer_num=bad_num
                )


if __name__ == "__main__":
    unittest.main()
