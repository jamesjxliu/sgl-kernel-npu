// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acl/acl.h"
#include "defines.h"
#include "torch_helper.h"

#include <algorithm>
#include <tuple>
#include <vector>

namespace sglang {
namespace npu_kernel {

constexpr int64_t KV_TRANS_FLAG_1D = 1 << 0;
constexpr int64_t KV_TRANS_FLAG_2D = 1 << 1;

enum TransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

// @direction: only support 1 or 2, 1 is H2D, 2 is D2H
// @flags: only support 2
// @layer_start: first layer to transfer (dim 0 of device_* / dim 1 of host_*).
//     Enables layer-group pipelining: callers split a full-buffer copy into
//     per-group copies so that per-layer completion events recorded between
//     consecutive calls fire progressively instead of only after the whole
//     transfer.
// @layer_num: number of layers to transfer; negative means all layers.
HOST_API void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v,
                                       at::Tensor &host_v, const at::Tensor &device_indices,
                                       const at::Tensor &host_indices, int64_t page_size, int64_t direction,
                                       int64_t flags, int64_t layer_start, int64_t layer_num)
{
    TORCH_CHECK(device_k.numel() != 0, "device_k must not be empty");
    TORCH_CHECK(host_k.numel() != 0, "host_k must not be empty");
    TORCH_CHECK(device_k.dim() == host_k.dim(), "the number of dimensions of device_k must be equal to host_k");
    TORCH_CHECK(device_k.dim() == 5, "the number of dimensions of device_k must be 5");
    TORCH_CHECK(device_k.sizes()[0] == host_k.sizes()[1], "the layer number of device_k must be equal to host_k");
    TORCH_CHECK(device_k.sizes()[2] == page_size, "the 3rd dimension of device_k must be equal to page size");
    TORCH_CHECK(host_k.sizes()[2] == page_size, "the 3rd dimension of host_k must be equal to page size");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "device indices size must be divisible by page size");
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                    direction == static_cast<int64_t>(TransferDirection::D2H),
                "direction must be equal to 1(h2d) or 2(d2h)")
    TORCH_CHECK((flags & KV_TRANS_FLAG_2D) == KV_TRANS_FLAG_2D, "now only support 2d(flags=2) copy");

    if (device_v.numel() != 0 && host_v.numel() != 0) {
        TORCH_CHECK(device_v.dim() == host_v.dim(), "the number of dimensions of device_v must be equal to host_v");
        TORCH_CHECK(device_v.dim() == 5, "the number of dimensions of device_v must be 5");
        TORCH_CHECK(device_v.sizes()[0] == host_v.sizes()[1], "the layer number of device_v must be equal to host_v");
        TORCH_CHECK(device_v.sizes()[2] == page_size, "the 3rd dimension of device_v must be equal to page size");
        TORCH_CHECK(host_v.sizes()[2] == page_size, "the 3rd dimension of host_v must be equal to page size");
    }

    auto device_indices_cpu = device_indices.cpu();
    auto host_indices_cpu = host_indices.cpu();
    const int64_t device_pages_num = device_k.sizes()[1];
    const int64_t host_pages_num = host_k.sizes()[0];
    const int64_t total_num_layers = device_k.sizes()[0];
    const int64_t height = layer_num < 0 ? total_num_layers : layer_num;
    TORCH_CHECK(layer_start >= 0, "layer_start must be non-negative");
    TORCH_CHECK(height > 0, "layer_num must be positive (or negative to transfer all layers)");
    TORCH_CHECK(layer_start + height <= total_num_layers,
                "layer_start + layer_num must not exceed the layer number of device_k");
    const auto heads_num = device_k.sizes()[3];
    const auto item_size = device_k.element_size();
    const auto k_head_dim = device_k.sizes()[4];
    const auto k_device_pitch = device_pages_num * page_size * heads_num * k_head_dim * item_size;
    const auto k_host_pitch = page_size * heads_num * k_head_dim * item_size;
    const auto k_width = page_size * heads_num * k_head_dim * item_size;
    // device_v may be an empty tensor (e.g. FP8 packed KV); guard the
    // sizes()[4] read so the pitches are only derived from a valid shape.
    const auto v_head_dim = (device_v.numel() != 0 && host_v.numel() != 0) ? device_v.sizes()[4] : 0;
    const auto v_device_pitch = device_pages_num * page_size * heads_num * v_head_dim * item_size;
    const auto v_host_pitch = page_size * heads_num * v_head_dim * item_size;
    const auto v_width = page_size * heads_num * v_head_dim * item_size;
    c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
    aclrtStream acl_stream = current_stream.stream();

    const int64_t num_pages = device_indices.size(0) / page_size;
    TORCH_CHECK(device_indices.scalar_type() == at::kLong, "device_indices must be int64");
    TORCH_CHECK(host_indices.scalar_type() == at::kLong, "host_indices must be int64");

    // ---- Hoist all loop-invariant addressing out of the page loop. ----
    // Per-page addresses are computed as base + offset from the base pointers
    // and element strides, avoiding per-page at::Tensor indexing (which
    // materializes temporary TensorImpls) and per-page .item() dispatches.
    // stride() * item_size also stays correct for sliced views (data_ptr()
    // already carries the storage offset), unlike offsets derived from sizes.
    const int64_t *device_idx = device_indices_cpu.data_ptr<int64_t>();
    const int64_t *host_idx = host_indices_cpu.data_ptr<int64_t>();

    // device_* layout: (layer, page, page_size, heads, head_dim)
    // host_* layout:  (page, layer, page_size, heads, head_dim)
    char *device_k_base = static_cast<char *>(device_k.data_ptr());
    char *host_k_base = static_cast<char *>(host_k.data_ptr());
    const int64_t device_k_layer_pitch = device_k.stride(0) * item_size;
    const int64_t device_k_page_pitch = device_k.stride(1) * item_size;
    const int64_t host_k_page_pitch = host_k.stride(0) * item_size;
    const int64_t host_k_layer_pitch = host_k.stride(1) * item_size;
    // The innermost three dims (page_size, heads, head_dim) must be one
    // contiguous block so that a 2D row of the memcpy is a flat copy.
    TORCH_CHECK(device_k.stride(1) == page_size * heads_num * k_head_dim,
                "device_k innermost dims must be contiguous");
    TORCH_CHECK(host_k.stride(1) == page_size * heads_num * k_head_dim,
                "host_k innermost dims must be contiguous");

    const bool has_v = device_v.numel() != 0 && host_v.numel() != 0;
    char *device_v_base = has_v ? static_cast<char *>(device_v.data_ptr()) : nullptr;
    char *host_v_base = has_v ? static_cast<char *>(host_v.data_ptr()) : nullptr;
    int64_t device_v_layer_pitch = 0, device_v_page_pitch = 0;
    int64_t host_v_page_pitch = 0, host_v_layer_pitch = 0;
    if (has_v) {
        device_v_layer_pitch = device_v.stride(0) * item_size;
        device_v_page_pitch = device_v.stride(1) * item_size;
        host_v_page_pitch = host_v.stride(0) * item_size;
        host_v_layer_pitch = host_v.stride(1) * item_size;
        TORCH_CHECK(device_v.stride(1) == page_size * heads_num * v_head_dim,
                    "device_v innermost dims must be contiguous");
        TORCH_CHECK(host_v.stride(1) == page_size * heads_num * v_head_dim,
                    "host_v innermost dims must be contiguous");
    }

    const bool is_d2h = direction == static_cast<int64_t>(TransferDirection::D2H);
    const aclrtMemcpyKind copy_kind =
        is_d2h ? aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST : aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;

    // ---- Page-run transfer with run-length merging. ----
    // Indices produced by the KV allocators are typically long contiguous runs
    // on both sides. For a run of n pages that is contiguous on host AND device
    // (host page p..p+n-1 and device page q..q+n-1), the per-page 2D copy
    // (height=layers, dst strides across layers) can be transposed into a
    // per-layer 2D copy (height=n, dst strides across contiguous device pages,
    // src strides across host pages). This is mathematically equivalent but:
    //   - reduces memcpy2d calls from n to `height` per run (e.g. 959 pages ->
    //     78 calls when fully contiguous),
    //   - makes the device side of each H2D task one large contiguous block,
    //     which is friendlier to the DMA engine than layer-strided rows.
    // Runs shorter than `height` fall back to the per-page form.
    int64_t i = 0;
    while (i < num_pages) {
        const int64_t device_page0 = device_idx[i * page_size] / page_size;
        const int64_t host_page0 = host_idx[i * page_size] / page_size;
        TORCH_CHECK(device_page0 >= 0 && device_page0 < device_pages_num,
                    "device_page_index must be less than the 2nd dim of device_k");
        TORCH_CHECK(host_page0 >= 0 && host_page0 < host_pages_num,
                    "host_page_index must be less than the 1st dim of host_k");

        int64_t run = 1;
        while (i + run < num_pages &&
               device_idx[(i + run) * page_size] == device_idx[(i + run - 1) * page_size] + page_size &&
               host_idx[(i + run) * page_size] == host_idx[(i + run - 1) * page_size] + page_size) {
            ++run;
        }
        // Page continuity implies the last page index is device_page0 + run - 1.
        TORCH_CHECK(device_page0 + run <= device_pages_num,
                    "device page run exceeds the 2nd dim of device_k");
        TORCH_CHECK(host_page0 + run <= host_pages_num,
                    "host page run exceeds the 1st dim of host_k");

        if (run >= height) {
            // Merged form: one 2D copy per layer, rows are pages.
            // dst rows (H2D) walk contiguous device pages (pitch = one page of
            // one layer); src rows walk host pages (pitch = one full host page
            // = total_num_layers rows).
            for (int64_t l = 0; l < height; ++l) {
                char *device_k_ptr = device_k_base + (layer_start + l) * device_k_layer_pitch +
                                     device_page0 * device_k_page_pitch;
                char *host_k_ptr = host_k_base + host_page0 * host_k_page_pitch +
                                   (layer_start + l) * host_k_layer_pitch;
                if (is_d2h) {
                    aclrtMemcpy2dAsync(host_k_ptr, host_k_page_pitch, device_k_ptr, device_k_page_pitch,
                                       k_width, run, copy_kind, acl_stream);
                } else {
                    aclrtMemcpy2dAsync(device_k_ptr, device_k_page_pitch, host_k_ptr, host_k_page_pitch,
                                       k_width, run, copy_kind, acl_stream);
                }

                if (has_v) {
                    char *device_v_ptr = device_v_base + (layer_start + l) * device_v_layer_pitch +
                                         device_page0 * device_v_page_pitch;
                    char *host_v_ptr = host_v_base + host_page0 * host_v_page_pitch +
                                       (layer_start + l) * host_v_layer_pitch;
                    if (is_d2h) {
                        aclrtMemcpy2dAsync(host_v_ptr, host_v_page_pitch, device_v_ptr, device_v_page_pitch,
                                           v_width, run, copy_kind, acl_stream);
                    } else {
                        aclrtMemcpy2dAsync(device_v_ptr, device_v_page_pitch, host_v_ptr, host_v_page_pitch,
                                           v_width, run, copy_kind, acl_stream);
                    }
                }
            }
        } else {
            // Per-page form (existing behavior): one 2D copy per page, rows are layers.
            for (int64_t j = 0; j < run; ++j) {
                const int64_t device_page_index = device_page0 + j;
                const int64_t host_page_index = host_page0 + j;

                char *device_k_ptr = device_k_base + layer_start * device_k_layer_pitch +
                                     device_page_index * device_k_page_pitch;
                char *host_k_ptr = host_k_base + host_page_index * host_k_page_pitch +
                                   layer_start * host_k_layer_pitch;
                if (is_d2h) {
                    aclrtMemcpy2dAsync(host_k_ptr, k_host_pitch, device_k_ptr, k_device_pitch, k_width, height,
                                       copy_kind, acl_stream);
                } else {
                    aclrtMemcpy2dAsync(device_k_ptr, k_device_pitch, host_k_ptr, k_host_pitch, k_width, height,
                                       copy_kind, acl_stream);
                }

                if (has_v) {
                    char *device_v_ptr = device_v_base + layer_start * device_v_layer_pitch +
                                         device_page_index * device_v_page_pitch;
                    char *host_v_ptr = host_v_base + host_page_index * host_v_page_pitch +
                                       layer_start * host_v_layer_pitch;
                    if (is_d2h) {
                        aclrtMemcpy2dAsync(host_v_ptr, v_host_pitch, device_v_ptr, v_device_pitch, v_width,
                                           height, copy_kind, acl_stream);
                    } else {
                        aclrtMemcpy2dAsync(device_v_ptr, v_device_pitch, host_v_ptr, v_host_pitch, v_width,
                                           height, copy_kind, acl_stream);
                    }
                }
            }
        }
        i += run;
    }
}

// ---------------------------------------------------------------------------
// transfer_kv_dim_exchange_table: flat (src, dst, len) entry-table builder
// for the Memfabric acc_offload AIV sparse-copy kernel.
//
// The page_first(host) <-> layer_first(device) transpose can only be
// expressed with flat copy entries at (page, layer) granularity, so one entry
// is emitted per (page, layer) row of every component (k/v/index_k/scale).
// Rows wider than 88KB are split: the AIV KV-copy kernel (selected when the
// entry count >= blockDim) performs one DataCopyPad per entry into an 88KB
// UB double-buffer without internal chunking, so an entry larger than the
// buffer would overflow UB.
//
// Layer ranges (layer-group pipelining): the k/v components cover
// [layer_start, layer_start + layer_num) of their layer space (layer_num < 0
// means all layers), the index_k/scale components cover
// [index_k_layer_start, index_k_layer_start + index_k_layer_num) of the
// separate indexer layer space (num < 0 means all, 0 skips the components).
// This lets the caller interleave table builds + sparse_copy launches per
// layer group so per-group completion events overlap DMA with compute.
//
// Returns (src_ptrs int64[N], dst_ptrs int64[N], lens int32[N],
// size int32[1]) on the device of device_k, ready for offload.sparse_copy.
// ---------------------------------------------------------------------------
namespace {

constexpr int64_t ASCENDC_ENTRY_SPLIT = 88 * 1024;  // UB double-buffer size

struct TableComponent {
    const char *device_base = nullptr;
    const char *host_base = nullptr;
    int64_t layers = 0;
    int64_t device_layer_pitch = 0;
    int64_t device_page_pitch = 0;
    int64_t host_page_pitch = 0;
    int64_t host_layer_pitch = 0;
    int64_t width = 0;  // bytes of one (page, layer) row
    int64_t lo = 0;     // first layer (inclusive) of the transfer range
    int64_t hi = 0;     // last layer (exclusive) of the transfer range
};

TableComponent make_table_component(const at::Tensor &device_t, const at::Tensor &host_t,
                                    int64_t page_size, const char *name)
{
    TORCH_CHECK(device_t.numel() != 0 && host_t.numel() != 0,
                name, " must be non-empty on both device and host sides");
    TORCH_CHECK(device_t.dim() == 5 && host_t.dim() == 5,
                name, " must be 5-D on both device and host sides");
    TORCH_CHECK(device_t.sizes()[0] == host_t.sizes()[1],
                "the layer number of device_", name, " must be equal to host_", name);
    TORCH_CHECK(device_t.sizes()[1] == host_t.sizes()[0],
                "the page number of device_", name, " must be equal to host_", name);
    TORCH_CHECK(device_t.sizes()[2] == page_size && host_t.sizes()[2] == page_size,
                "the 3rd dimension of ", name, " must be equal to page size");
    TORCH_CHECK(device_t.sizes()[3] == host_t.sizes()[3],
                "the head number of device_", name, " must be equal to host_", name);

    const int64_t heads = device_t.sizes()[3];
    const int64_t head_dim = device_t.sizes()[4];
    const int64_t item_size = device_t.element_size();
    const int64_t width = page_size * heads * head_dim * item_size;
    TORCH_CHECK(device_t.stride(1) == page_size * heads * head_dim,
                "device_", name, " innermost dims must be contiguous");
    TORCH_CHECK(host_t.stride(1) == page_size * heads * head_dim,
                "host_", name, " innermost dims must be contiguous");

    TableComponent c;
    c.device_base = static_cast<const char *>(device_t.data_ptr());
    c.host_base = static_cast<const char *>(host_t.data_ptr());
    c.layers = device_t.sizes()[0];
    c.device_layer_pitch = device_t.stride(0) * item_size;
    c.device_page_pitch = device_t.stride(1) * item_size;
    c.host_page_pitch = host_t.stride(0) * item_size;
    c.host_layer_pitch = host_t.stride(1) * item_size;
    c.width = width;
    return c;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
transfer_kv_dim_exchange_table(const at::Tensor &device_k, const at::Tensor &host_k,
                               const at::Tensor &device_v, const at::Tensor &host_v,
                               const at::Tensor &device_index_k, const at::Tensor &host_index_k,
                               const at::Tensor &device_index_k_scale, const at::Tensor &host_index_k_scale,
                               const at::Tensor &device_indices, const at::Tensor &host_indices,
                               int64_t page_size, int64_t direction,
                               int64_t layer_start, int64_t layer_num,
                               int64_t index_k_layer_start, int64_t index_k_layer_num)
{
    TORCH_CHECK(device_k.numel() != 0, "device_k must not be empty");
    TORCH_CHECK(host_k.numel() != 0, "host_k must not be empty");
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                    direction == static_cast<int64_t>(TransferDirection::D2H),
                "direction must be equal to 1(h2d) or 2(d2h)")
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(),
                "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0,
                "device indices size must be divisible by page size");
    TORCH_CHECK(device_indices.scalar_type() == at::kLong, "device_indices must be int64");
    TORCH_CHECK(host_indices.scalar_type() == at::kLong, "host_indices must be int64");

    // Resolve the k/v transfer range [kv_lo, kv_hi) in the k/v layer space.
    auto kv = make_table_component(device_k, host_k, page_size, "k");
    const int64_t kv_lo = layer_start;
    const int64_t kv_hi = layer_num < 0 ? kv.layers : layer_start + layer_num;
    TORCH_CHECK(kv_lo >= 0 && kv_lo < kv_hi && kv_hi <= kv.layers,
                "k/v layer range [", kv_lo, ", ", kv_hi, ") is invalid for ",
                kv.layers, " layers");

    // Resolve the index_k/scale range in the separate indexer layer space.
    // index_k_layer_num == 0 skips those components (the k/v group contains
    // no indexer layers); a negative value means the whole space.
    const bool has_index_k = index_k_layer_num != 0 && device_index_k.numel() != 0 &&
                             host_index_k.numel() != 0;
    const bool has_scale = index_k_layer_num != 0 && device_index_k_scale.numel() != 0 &&
                           host_index_k_scale.numel() != 0;
    int64_t ik_lo = 0, ik_hi = 0;
    if (has_index_k || has_scale) {
        const at::Tensor &probe = has_index_k ? device_index_k : device_index_k_scale;
        const int64_t ik_layers = probe.sizes()[0];
        ik_lo = index_k_layer_start;
        ik_hi = index_k_layer_num < 0 ? ik_layers : index_k_layer_start + index_k_layer_num;
        TORCH_CHECK(ik_lo >= 0 && ik_lo < ik_hi && ik_hi <= ik_layers,
                    "index_k layer range [", ik_lo, ", ", ik_hi, ") is invalid for ",
                    ik_layers, " layers");
    }

    std::vector<TableComponent> components;
    components.reserve(4);
    kv.lo = kv_lo;
    kv.hi = kv_hi;
    components.push_back(kv);
    if (device_v.numel() != 0 && host_v.numel() != 0) {
        TableComponent v = make_table_component(device_v, host_v, page_size, "v");
        v.lo = kv_lo;
        v.hi = kv_hi;
        components.push_back(v);
    }
    if (has_index_k) {
        TableComponent ik = make_table_component(device_index_k, host_index_k, page_size, "index_k");
        ik.lo = ik_lo;
        ik.hi = ik_hi;
        components.push_back(ik);
    }
    if (has_scale) {
        TableComponent sc = make_table_component(device_index_k_scale, host_index_k_scale,
                                                 page_size, "index_k_scale");
        sc.lo = ik_lo;
        sc.hi = ik_hi;
        components.push_back(sc);
    }

    auto device_indices_cpu = device_indices.cpu();
    auto host_indices_cpu = host_indices.cpu();
    const int64_t *device_idx = device_indices_cpu.data_ptr<int64_t>();
    const int64_t *host_idx = host_indices_cpu.data_ptr<int64_t>();
    const int64_t num_pages = device_indices.size(0) / page_size;

    // Bounds-check pages against the k component (all components share the
    // page count; make_table_component already asserts matching page numbers).
    const int64_t device_pages_num = device_k.sizes()[1];
    const int64_t host_pages_num = host_k.sizes()[0];

    int64_t count = 0;
    for (const auto &c : components) {
        const int64_t nsplit = (c.width + ASCENDC_ENTRY_SPLIT - 1) / ASCENDC_ENTRY_SPLIT;
        count += num_pages * (c.hi - c.lo) * nsplit;
    }

    // Build the table in pinned host memory, then upload async on the current
    // stream (at::copy_ from pinned CPU records a stream event in the NPU
    // caching host allocator, so the staging tensors can die at return).
    auto pinned = at::TensorOptions().pinned_memory(true);
    auto src_cpu = at::empty({count}, pinned.dtype(at::kLong));
    auto dst_cpu = at::empty({count}, pinned.dtype(at::kLong));
    auto len_cpu = at::empty({count}, pinned.dtype(at::kInt));
    auto size_cpu = at::empty({1}, pinned.dtype(at::kInt));

    int64_t *src_p = src_cpu.data_ptr<int64_t>();
    int64_t *dst_p = dst_cpu.data_ptr<int64_t>();
    int32_t *len_p = len_cpu.data_ptr<int32_t>();
    const bool is_d2h = direction == static_cast<int64_t>(TransferDirection::D2H);

    int64_t e = 0;
    for (int64_t i = 0; i < num_pages; ++i) {
        const int64_t device_page = device_idx[i * page_size] / page_size;
        const int64_t host_page = host_idx[i * page_size] / page_size;
        TORCH_CHECK(device_page >= 0 && device_page < device_pages_num,
                    "device_page_index must be less than the 2nd dim of device_k");
        TORCH_CHECK(host_page >= 0 && host_page < host_pages_num,
                    "host_page_index must be less than the 1st dim of host_k");
        for (const auto &c : components) {
            for (int64_t l = c.lo; l < c.hi; ++l) {
                const char *device_row =
                    c.device_base + l * c.device_layer_pitch + device_page * c.device_page_pitch;
                const char *host_row =
                    c.host_base + host_page * c.host_page_pitch + l * c.host_layer_pitch;
                const char *sp = is_d2h ? device_row : host_row;
                const char *dp = is_d2h ? host_row : device_row;
                for (int64_t off = 0; off < c.width; off += ASCENDC_ENTRY_SPLIT) {
                    const int64_t len = std::min(ASCENDC_ENTRY_SPLIT, c.width - off);
                    src_p[e] = reinterpret_cast<int64_t>(sp + off);
                    dst_p[e] = reinterpret_cast<int64_t>(dp + off);
                    len_p[e] = static_cast<int32_t>(len);
                    ++e;
                }
            }
        }
    }
    TORCH_CHECK(e == count, "entry count mismatch during table construction");
    *size_cpu.data_ptr<int32_t>() = static_cast<int32_t>(count);

    auto out_opts = device_k.options();
    auto src_out = at::empty({count}, out_opts.dtype(at::kLong));
    auto dst_out = at::empty({count}, out_opts.dtype(at::kLong));
    auto len_out = at::empty({count}, out_opts.dtype(at::kInt));
    auto size_out = at::empty({1}, out_opts.dtype(at::kInt));
    src_out.copy_(src_cpu, /*non_blocking=*/true);
    dst_out.copy_(dst_cpu, /*non_blocking=*/true);
    len_out.copy_(len_cpu, /*non_blocking=*/true);
    size_out.copy_(size_cpu, /*non_blocking=*/true);
    return std::make_tuple(std::move(src_out), std::move(dst_out), std::move(len_out),
                           std::move(size_out));
}

}  // namespace npu_kernel
}  // namespace sglang
