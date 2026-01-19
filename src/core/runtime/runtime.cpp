#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    _api = llaisys::device::getRuntimeAPI(_device_type);
    _stream = _api->create_stream();
    _allocator = new allocators::NaiveAllocator(_api);
}

Runtime::~Runtime() {
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

/**
 * ----
 * 返回 runtime 使用的底层 API 指针
 * 例如 CUDA runtime 会返回 CUDARuntimeAPI 结构体指针
 */
const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)_api->malloc_host(size), size, *this, true));
    // 注意这里强制转换为 std::byte*，因为 malloc_host 返回的是 void*
}

void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
