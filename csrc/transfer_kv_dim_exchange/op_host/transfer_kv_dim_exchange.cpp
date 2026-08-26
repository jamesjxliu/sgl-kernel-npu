// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

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

namespace {

void check_acl_copy(aclError result, const char *direction, size_t width, size_t height)
{
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtMemcpy2dAsync failed for kv ", direction,
                " transfer: error=", static_cast<int64_t>(result), ", width=", width, ", height=", height);
}

// Validate that the (page_size, heads, head_dim) trailing payload of a 5-D KV
// tensor is physically dense so aclrtMemcpy2dAsync can treat one page as a
// single contiguous row.  Returns the slot byte size.
size_t validate_dense_slot_payload(const at::Tensor &tensor, int64_t component)
{
    // tensor is [layers_or_pages, pages_or_layers, page_size, heads, head_dim]
    std::vector<std::pair<int64_t, int64_t>> payload_dims;
    int64_t slot_elements = 1;
    for (int64_t dim = 2; dim < tensor.dim(); ++dim) {
        const int64_t size = tensor.size(dim);
        const int64_t stride = tensor.stride(dim);
        TORCH_CHECK(stride >= 0, "kv component ", component, " has a negative stride at dimension ", dim);
        slot_elements *= size;
        if (size > 1) {
            payload_dims.emplace_back(stride, size);
        }
    }

    std::sort(payload_dims.begin(), payload_dims.end());
    int64_t expected_stride = 1;
    for (const auto &[stride, size] : payload_dims) {
        TORCH_CHECK(stride == expected_stride, "kv component ", component,
                    " slot payload must be physically dense; got payload stride ", stride, " while expecting ",
                    expected_stride);
        expected_stride *= size;
    }
    return static_cast<size_t>(slot_elements) * tensor.element_size();
}

// Device layout: [layers, pages, page_size, heads, head_dim]   (layer-first)
// Host   layout: [pages, layers, page_size, heads, head_dim]   (page-first)
// Both must share the same (page_size, heads, head_dim) trailing payload.
struct KVComponentLayout {
    at::Tensor device;
    at::Tensor host;
    int64_t device_page_num;
    int64_t host_page_num;
    int64_t layer_num;
    size_t slot_bytes;          // bytes of one page (page_size*heads*head_dim*itemsize)
    size_t device_layer_pitch;  // bytes between consecutive layers on device
    size_t device_page_pitch;   // bytes between consecutive pages on device (same layer)
    size_t host_page_pitch;     // bytes between consecutive pages on host (same layer)
    size_t host_layer_pitch;    // bytes between consecutive layers on host
};

KVComponentLayout validate_kv_component(const at::Tensor &device, const at::Tensor &host, int64_t component,
                                        int64_t page_size)
{
    TORCH_CHECK(device.defined() && host.defined(), "kv component ", component, " must be defined");
    TORCH_CHECK(device.numel() != 0, "device kv component ", component, " must not be empty");
    TORCH_CHECK(host.numel() != 0, "host kv component ", component, " must not be empty");
    TORCH_CHECK(device.device().type() == c10::DeviceType::PrivateUse1, "device kv component ", component,
                " must be on NPU, got ", device.device());
    TORCH_CHECK(host.device().is_cpu(), "host kv component ", component, " must be on CPU, got ", host.device());
    TORCH_CHECK(device.scalar_type() == host.scalar_type(), "kv component ", component,
                " has different device/host dtypes: ", device.scalar_type(), " vs ", host.scalar_type());
    TORCH_CHECK(device.dim() == 5, "device kv component ", component,
                " must be 5-D [layers, pages, page_size, heads, head_dim], got ", device.dim(), " dimensions");
    TORCH_CHECK(host.dim() == 5, "host kv component ", component,
                " must be 5-D [pages, layers, page_size, heads, head_dim], got ", host.dim(), " dimensions");
    TORCH_CHECK(device.size(0) == host.size(1), "kv component ", component,
                " layer count mismatch: device=", device.size(0), " host=", host.size(1));
    TORCH_CHECK(device.size(1) == host.size(0), "kv component ", component,
                " page count mismatch: device=", device.size(1), " host=", host.size(0));
    TORCH_CHECK(device.size(2) == page_size, "kv component ", component,
                " device page_size=", device.size(2), " expected=", page_size);
    TORCH_CHECK(host.size(2) == page_size, "kv component ", component,
                " host page_size=", host.size(2), " expected=", page_size);
    TORCH_CHECK(device.size(3) == host.size(3), "kv component ", component, " heads mismatch");
    TORCH_CHECK(device.size(4) == host.size(4), "kv component ", component, " head_dim mismatch");

    const size_t slot_bytes = validate_dense_slot_payload(device, component);
    TORCH_CHECK(validate_dense_slot_payload(host, component) == slot_bytes,
                "kv component ", component, " device/host slot byte sizes differ");

    const auto item_size = device.element_size();
    const size_t device_layer_pitch = static_cast<size_t>(device.stride(0)) * item_size;
    const size_t device_page_pitch = static_cast<size_t>(device.stride(1)) * item_size;
    const size_t host_page_pitch = static_cast<size_t>(host.stride(0)) * item_size;
    const size_t host_layer_pitch = static_cast<size_t>(host.stride(1)) * item_size;
    TORCH_CHECK(slot_bytes <= device_layer_pitch && slot_bytes <= device_page_pitch &&
                    slot_bytes <= host_page_pitch && slot_bytes <= host_layer_pitch,
                "invalid kv component ", component, " pitch for aclrtMemcpy2dAsync");
    return {
        device,
        host,
        device.size(1),
        host.size(0),
        device.size(0),
        slot_bytes,
        device_layer_pitch,
        device_page_pitch,
        host_page_pitch,
        host_layer_pitch,
    };
}

void validate_page_indices(const int64_t *device_pages, const int64_t *host_pages, int64_t count,
                           int64_t device_page_limit, int64_t host_page_limit, int64_t component)
{
    for (const auto i : c10::irange(count)) {
        TORCH_CHECK(device_pages[i] >= 0, "device page ", device_pages[i], " must be non-negative");
        TORCH_CHECK(host_pages[i] >= 0, "host page ", host_pages[i], " must be non-negative");
        TORCH_CHECK(device_pages[i] < device_page_limit, "device page ", device_pages[i],
                    " exceeds kv component ", component, " page count ", device_page_limit);
        TORCH_CHECK(host_pages[i] < host_page_limit, "host page ", host_pages[i],
                    " exceeds kv component ", component, " page count ", host_page_limit);
    }
}

// Convert token indices (one per token, page_size tokens per page) into page
// indices (one per page).  Only the first token of each page is read since all
// tokens in a page share the same page index.  Using bulk data_ptr<int64_t>
// access instead of per-page item<int64_t>() avoids host synchronization.
std::vector<int64_t> extract_page_indices(const at::Tensor &indices, int64_t num_pages, int64_t page_size)
{
    std::vector<int64_t> pages;
    pages.reserve(num_pages);
    const auto *token_indices = indices.data_ptr<int64_t>();
    for (const auto i : c10::irange(num_pages)) {
        pages.push_back(token_indices[i * page_size] / page_size);
    }
    return pages;
}

// Group consecutive (device_page, host_page) pairs that advance in lock-step
// into a single contiguous run.  Each run becomes one aclrtMemcpy2dAsync call
// with height=run_length, drastically reducing DMA dispatches for per-layer
// copies where the slot-major layout makes pages physically adjacent.
std::vector<std::pair<int64_t, int64_t>> build_contiguous_runs(const int64_t *device_pages,
                                                               const int64_t *host_pages, int64_t count)
{
    std::vector<std::pair<int64_t, int64_t>> runs;
    runs.reserve(count);
    int64_t begin = 0;
    while (begin < count) {
        int64_t end = begin + 1;
        while (end < count && device_pages[end] == device_pages[end - 1] + 1 &&
               host_pages[end] == host_pages[end - 1] + 1) {
            ++end;
        }
        runs.emplace_back(begin, end - begin);
        begin = end;
    }
    return runs;
}

struct PreparedTransfer {
    std::vector<KVComponentLayout> components;
    std::vector<int64_t> device_pages;
    std::vector<int64_t> host_pages;
    int64_t num_pages = 0;
    aclrtStream stream = nullptr;
    bool is_d2h = false;
    aclrtMemcpyKind copy_kind = ACL_MEMCPY_DEVICE_TO_HOST;
};

PreparedTransfer prepare_transfer(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v,
                                  at::Tensor &host_v, const at::Tensor &device_indices,
                                  const at::Tensor &host_indices, int64_t page_size, int64_t direction,
                                  int64_t flags)
{
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                    direction == static_cast<int64_t>(TransferDirection::D2H),
                "direction must be equal to 1(h2d) or 2(d2h)");
    TORCH_CHECK((flags & KV_TRANS_FLAG_2D) == KV_TRANS_FLAG_2D, "now only support 2d(flags=2) copy");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(),
                "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0,
                "device indices size must be divisible by page size");

    PreparedTransfer state;
    state.is_d2h = (direction == static_cast<int64_t>(TransferDirection::D2H));
    state.copy_kind = state.is_d2h ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
    state.stream = c10_npu::getCurrentNPUStream().stream();

    const auto device_indices_cpu = device_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const auto host_indices_cpu = host_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    state.num_pages = device_indices_cpu.numel() / page_size;
    if (state.num_pages == 0) {
        return state;
    }
    state.device_pages = extract_page_indices(device_indices_cpu, state.num_pages, page_size);
    state.host_pages = extract_page_indices(host_indices_cpu, state.num_pages, page_size);

    state.components.emplace_back(validate_kv_component(device_k, host_k, 0, page_size));
    if (device_v.numel() != 0 && host_v.numel() != 0) {
        state.components.emplace_back(validate_kv_component(device_v, host_v, 1, page_size));
    }
    validate_page_indices(state.device_pages.data(), state.host_pages.data(), state.num_pages,
                          state.components.front().device_page_num, state.components.front().host_page_num, 0);
    return state;
}

// Shared dispatch core: transfers the layer range [layer_start,
// layer_start + layer_num) of every component, where layer_num < 0 means all
// layers.  Page indices are grouped into contiguous runs; a run of length r
// is copied with min(r, height) aclrtMemcpy2dAsync dispatches (height = the
// component's transferred layer count):
//   - r >= height (merged form): one copy per layer, rows are pages
//     (height=r); the H2D destination rows walk contiguous device pages of
//     one single layer, which is friendlier to the DMA engine than
//     layer-strided rows;
//   - r < height (per-page form): one copy per page, rows are layers
//     (height=height).
// Contiguous page indices therefore drop the dispatch count from num_pages
// to the layer count per component.
void dispatch_kv_transfer(const PreparedTransfer &state, int64_t layer_start, int64_t layer_num)
{
    for (const auto &component : state.components) {
        const int64_t height = layer_num < 0 ? component.layer_num : layer_num;
        TORCH_CHECK(layer_start >= 0, "layer_start must be non-negative");
        TORCH_CHECK(height > 0, "layer_num must be positive (or negative to transfer all layers)");
        TORCH_CHECK(layer_start + height <= component.layer_num, "layer range [", layer_start, ", ",
                    layer_start + height, ") exceeds kv component layer count ", component.layer_num);
    }

    const auto runs = build_contiguous_runs(state.device_pages.data(), state.host_pages.data(), state.num_pages);
    for (const auto &component : state.components) {
        auto *device_base = static_cast<char *>(component.device.data_ptr());
        auto *host_base = static_cast<char *>(component.host.data_ptr());
        const int64_t height = layer_num < 0 ? component.layer_num : layer_num;
        for (const auto &[run_begin, run_length] : runs) {
            if (run_length >= height) {
                // Merged form: one 2D copy per layer, rows are pages.  H2D dst
                // rows walk contiguous device pages within one layer (pitch =
                // one page); src rows walk host pages (pitch = one full host
                // page = all layers).
                const auto device_page0 = static_cast<size_t>(state.device_pages[run_begin]);
                const auto host_page0 = static_cast<size_t>(state.host_pages[run_begin]);
                TORCH_CHECK(device_page0 + static_cast<size_t>(run_length) <=
                                static_cast<size_t>(component.device_page_num),
                            "device page run exceeds the page count of device_k");
                TORCH_CHECK(host_page0 + static_cast<size_t>(run_length) <=
                                static_cast<size_t>(component.host_page_num),
                            "host page run exceeds the page count of host_k");
                const size_t copy_height = static_cast<size_t>(run_length);
                for (const auto layer : c10::irange(height)) {
                    void *device_ptr = device_base + static_cast<size_t>(layer_start + layer) *
                                                          component.device_layer_pitch +
                                       device_page0 * component.device_page_pitch;
                    void *host_ptr = host_base + host_page0 * component.host_page_pitch +
                                     static_cast<size_t>(layer_start + layer) * component.host_layer_pitch;
                    aclError result;
                    if (state.is_d2h) {
                        result = aclrtMemcpy2dAsync(host_ptr, component.host_page_pitch, device_ptr,
                                                    component.device_page_pitch, component.slot_bytes, copy_height,
                                                    state.copy_kind, state.stream);
                    } else {
                        result = aclrtMemcpy2dAsync(device_ptr, component.device_page_pitch, host_ptr,
                                                    component.host_page_pitch, component.slot_bytes, copy_height,
                                                    state.copy_kind, state.stream);
                    }
                    check_acl_copy(result, state.is_d2h ? "D2H" : "H2D", component.slot_bytes, copy_height);
                }
            } else {
                // Per-page form: one 2D copy per page, rows are layers
                // (height rows starting at layer_start).
                const size_t layer_offset_device = static_cast<size_t>(layer_start) * component.device_layer_pitch;
                const size_t layer_offset_host = static_cast<size_t>(layer_start) * component.host_layer_pitch;
                for (const auto j : c10::irange(run_length)) {
                    const auto device_page = state.device_pages[run_begin + j];
                    const auto host_page = state.host_pages[run_begin + j];
                    void *device_ptr = device_base + device_page * component.device_page_pitch + layer_offset_device;
                    void *host_ptr = host_base + host_page * component.host_page_pitch + layer_offset_host;
                    aclError result;
                    if (state.is_d2h) {
                        result = aclrtMemcpy2dAsync(host_ptr, component.host_layer_pitch, device_ptr,
                                                    component.device_layer_pitch, component.slot_bytes,
                                                    static_cast<size_t>(height), state.copy_kind, state.stream);
                    } else {
                        result = aclrtMemcpy2dAsync(device_ptr, component.device_layer_pitch, host_ptr,
                                                    component.host_layer_pitch, component.slot_bytes,
                                                    static_cast<size_t>(height), state.copy_kind, state.stream);
                    }
                    check_acl_copy(result, state.is_d2h ? "D2H" : "H2D", component.slot_bytes,
                                   static_cast<size_t>(height));
                }
            }
        }
    }
}

}  // namespace

// KV dimension-exchange transfer for the layer range [layer_start,
// layer_start + layer_num) of each component, where layer_num < 0 means all
// layers and layer_num = 1 degenerates to the historical per-layer form.
// Forwards to the shared dispatch core; see dispatch_kv_transfer for the
// adaptive min(run, layer_num) dispatch strategy.
//
// @layer_start: first layer to transfer, in the component's own layer space.
// @layer_num: number of layers to transfer; negative means all layers.
// @direction: only support 1 or 2, 1 is H2D, 2 is D2H
// @flags: only support 2
HOST_API void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v,
                                       at::Tensor &host_v, const at::Tensor &device_indices,
                                       const at::Tensor &host_indices, int64_t page_size, int64_t layer_start,
                                       int64_t layer_num, int64_t direction, int64_t flags)
{
    auto state = prepare_transfer(device_k, host_k, device_v, host_v, device_indices, host_indices, page_size,
                                   direction, flags);
    if (state.num_pages == 0) {
        return;
    }
    dispatch_kv_transfer(state, layer_start, layer_num);
}

}  // namespace npu_kernel
}  // namespace sglang
