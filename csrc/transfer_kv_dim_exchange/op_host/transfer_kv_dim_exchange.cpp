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

    for (int64_t i = 0; i < num_pages; ++i) {
        const int64_t device_page_index = device_idx[i * page_size] / page_size;
        const int64_t host_page_index = host_idx[i * page_size] / page_size;
        TORCH_CHECK(device_page_index >= 0 && device_page_index < device_pages_num,
                    "device_page_index must be less than the 2nd dim of device_k");
        TORCH_CHECK(host_page_index >= 0 && host_page_index < host_pages_num,
                    "host_page_index must be less than the 1st dim of host_k");

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
                aclrtMemcpy2dAsync(host_v_ptr, v_host_pitch, device_v_ptr, v_device_pitch, v_width, height,
                                   copy_kind, acl_stream);
            } else {
                aclrtMemcpy2dAsync(device_v_ptr, v_device_pitch, host_v_ptr, v_host_pitch, v_width, height,
                                   copy_kind, acl_stream);
            }
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
