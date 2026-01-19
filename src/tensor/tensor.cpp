#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size(); // 计算维度数量
    // 计算默认的行主序 strides
    std::vector<ptrdiff_t> strides(ndim_); // ptrdiff_t 实际是用于表示指针差值的有符号整数类型
    size_t stride = 1; // 初始化为1，表示最后一个维度的跨度为1
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i]; // 往前每一维，跨度乘以该维度的大小（例如shape = {2,3,4}, strides = {12,4,1}）
    }
    TensorMeta meta{dtype, shape, strides}; // 于是得到了 TensorMeta 信息
    size_t total_elems = stride; // total_elems 是所有维度大小的乘积
    size_t dtype_size = utils::dsize(dtype); // dtype_size 是数据类型的字节大小

    // 分配存储，这里的逻辑其实就是如果请求创建 CPU tensor，但当前 runtime 不是 CPU，则临时切换到 CPU 分配内存，如果请求创建的不是 CPU tensor，则切换到请求的设备类型分配内存。在创建的时候分别调用不同 runtime 下的 allocateHostStorage 或 allocateDeviceStorage 方法。
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        // 如果当前 runtime 不是 CPU，但请求创建 CPU tensor，则临时切换到 CPU 以分配内存
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size); // 从当前 runtime 分配 total_elems * dtype_size 字节的主机内存
        return std::shared_ptr<Tensor>(new Tensor(meta, storage)); // 返回的是一个 Tensor 智能指针，而storage 是 shared_ptr<Storage>，也是智能指针
    } else {
        // 否则，切换到请求的设备类型以分配内存
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

/**
 * ----
 * 返回 tensor 的实际数据位置 = 仓库起始位置 + 偏移量。
 */
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

/**
 * ----
 * 返回 tensor 的实际数据位置 = 仓库起始位置 + 偏移量。（常量版本，只读）
 */
const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

/**
 * ----
 * 返回 tensor 的维度数量
 */
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

/**
 * ----
 * 返回 tensor 的 strides 信息
 */
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

/**
 * ----
 * 计算并返回 tensor 的元素总数
 * 使用 std::accumulate 来对 shape 向量中的所有维度大小进行累积乘法，初始值为1。
 */
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

/**
 * ----
 * 返回 tensor 的信息字符串，包括形状、strides 和数据类型
 * 打印的时候使用 stringstream 来构建字符串
 * 会打印如下格式的信息：
 * Tensor: shape[ 2 3 4 ] strides[ 12 4 1 ] dtype=13
 */
std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    // 递归打印多维数组数据，如果dim等于最后一个维度，则打印该维度的数据，否则递归进入下一个维度
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

/**
 * ----
 * 调试打印 tensor 的信息和数据内容
 * 先打印 info 信息，然后根据设备类型决定如何打印数据内容。
 * 如果是 CPU 设备，直接打印数据；如果是其他设备，先将数据拷贝到临时的 CPU tensor，再打印。
 */
void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    // 1. 打印 tensor 信息
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        // 先创建一个临时的 CPU tensor
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        // 然后从当前设备拷贝数据到 CPU tensor（同步拷贝）
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

/**
 * ----
 * 判断 tensor 是否是连续存储的
 * 核心是检查 strides 与 shape 之间的关系，确保每个维度的 stride 等于后续维度的元素总数乘以数据类型大小。
 */
bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1;
    for (int i = ndim() - 1; i >= 0; i--) {
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= _meta.shape[i];
    }
    return true;
}

/**
 * ----
 * 返回一个新的 tensor，该 tensor 通过重新排列当前 tensor 的维度来实现维度置换。
 * 需要做的工作是根据给定的 order 计算新的形状和 strides 信息。
 * 但首先需要检查 order 的合法性：
 * 1. order 的长度必须等于当前 tensor 的维度数量，否则抛出异常
 * 2. order 中的每个值必须是有效的维度索引，且不能重复，否则抛出异常
 */
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    // 检查 order 的长度是否匹配
    if (order.size() != ndim_) {
        throw std::runtime_error("RuntimeError: Permute order size does not match tensor ndim.");
    }
    // 检查 order 中的值是否合法且不重复
    std::vector<bool> seen(ndim_, false);
    for (size_t dim : order) {
        if (dim >= ndim_ || seen[dim]) {
            throw std::runtime_error("RuntimeError: Invalid or duplicate dimension in permute order.");
        }
        seen[dim] = true;
    }
    // 计算新的 shape 和 strides 信息
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    // 创建新的元信息
    TensorMeta new_meta = _meta;
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));   
}

/**
 * ----
 * 返回一个新的 tensor，该 tensor 共享当前 tensor 的存储，但具有不同的形状视图。
 * 需要做的工作是计算新的 strides 信息以匹配新的形状。
 * 但首先需要检查
 * 1. 新形状的元素总数必须与当前 tensor 相同，否则抛出异常
 * 2. 内存必须是连续的，否则抛出异常
 */
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查新形状的元素总数是否匹配
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("RuntimeError: View shape does not match number of elements.");
    }

    // 检查内存是否连续
    if (!this->isContiguous()) {
        throw std::runtime_error("RuntimeError: view size is not compatible with input tensor's layout and stride... use .contiguous() first.");
    }

    // 计算新的 strides 信息，如果内存连续，其实新的 strides 也是行主序的计算方式
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    size_t stride = 1;
    for (size_t i = ndim_; i-- > 0;) {
        new_strides[i] = stride;
        stride *= shape[i];
    }
    // 创建新的元信息
    TensorMeta new_meta = _meta;
    new_meta.shape = shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

/**
 * ----
 * 返回一个新的 tensor，该 tensor 是当前 tensor 在指定维度上的切片。
 * 需要做的工作是计算新的形状和 strides 信息，以及新的偏移量。
 * 但首先需要检查切片参数的合法性：
 * 1. dim 必须是有效的维度索引，否则抛出异常
 * 2. start 和 end 必须在该维度的范围内，否则抛出异常
 */
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim_ = this->ndim();
    // 检查 dim 是否有效
    if (dim >= ndim_) {
        throw std::runtime_error("RuntimeError: Slice dimension out of range.");
    }
    // 检查 start 和 end 是否有效
    if (start >= end || end > _meta.shape[dim]) {
        throw std::runtime_error("RuntimeError: Invalid slice range.");
    }
    // 计算新的形状信息，其实就是将指定维度的大小改为 end - start
    std::vector<size_t> new_shape = _meta.shape;;
    new_shape[dim] = end - start;
    // 计算新的偏移量
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    // 创建新的元信息
    TensorMeta new_meta = _meta;
    new_meta.shape = new_shape;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

/**
 * ----
 * 从主机内存加载数据到 tensor 中
 * 注意这里的 src_ 是一个通用的 void* 指针，表示任意类型的数据源。
 * 另外，需要获取当前设备的 runtime API 来执行内存拷贝操作。
 */
void Tensor::load(const void *src_) {
    // 获取当前设备的 runtime API
    const LlaisysRuntimeAPI *api = core::context().runtime().api();
    // 计算要拷贝的数据总字节数
    size_t total_bytes = this->numel() * this->elementSize();
    // 执行内存拷贝操作，从主机内存 src_ 拷贝到 tensor 的数据存储位置
    api->memcpy_sync(
        this->data(),          // 目标地址，tensor 的数据存储位置
        src_,                  // 源地址，主机内存数据源
        total_bytes,          // 拷贝的字节数
        LLAISYS_MEMCPY_H2D); // 拷贝方向，主机到设备    
}

/**
 * ----
 * 返回一个新的 tensor，该 tensor 在内存中是连续存储的。
 * 如果当前 tensor 已经是连续的，则直接返回自身的共享指针。
 * 否则，创建一个新的连续存储的 tensor，并将当前 tensor 的数据拷贝过去。
 */
tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::const_pointer_cast<Tensor>(shared_from_this());
    }

    // 1. 创建一个新的连续存储的 tensor
    auto new_tensor = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());

    // 2. 根据设备类型，执行数据拷贝
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        this->_copyStridedOnCPU(new_tensor.get());
        return new_tensor;
    } else {
        throw std::runtime_error("Device-side contiguous() requires a custom kernel implementation.");
    }
}

/**
 * ----
 * 重塑 tensor 的形状
 */
tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (!this->isContiguous()) {
        // 如果当前 tensor 不是连续的，先转换为连续的 tensor
        return this->contiguous()->reshape(shape);
    }
    return this->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

void Tensor::_copyStridedOnCPU(Tensor* dst_tensor) const {
    // 首先获取api
    const LlaisysRuntimeAPI* api = core::context().runtime().api();

    // 计算维度和元素大小
    size_t ndim_ = this->ndim();
    size_t elem_size = this->elementSize();

    // 获取源数据的元数据
    const auto& src_shape = this->shape();
    const auto& src_strides = this->strides();
    const std::byte* src_base_ptr = reinterpret_cast<const std::byte*>(this->data());
    std::byte* dst_base_ptr = reinterpret_cast<std::byte*>(dst_tensor->data());

    // 优化：判断最内层维度是否连续，如果是，则可以一次性拷贝整个块
    bool inner_dim_contiguous = (ndim_ > 0) && (src_strides.back() == 1);

    // 定义块的大小，一个块就是一行，否则一个块就是一个元素
    size_t block_elem_count = inner_dim_contiguous ? src_shape.back() : 1;
    size_t block_byte_size = block_elem_count * elem_size;

    // 计算需要循环的维度数量
    size_t loop_dims = inner_dim_contiguous ? ndim_ - 1 : ndim_;

    // 如果是0维tensor或者完全连续的1维tensor，直接拷贝
    if (loop_dims == 0) {
        api->memcpy_sync(
            dst_base_ptr,
            src_base_ptr,
            block_byte_size,
            LLAISYS_MEMCPY_D2D);
        return;
    }

    // 准备循环索引
    std::vector<size_t> indices(loop_dims, 0);

    // 计算总的迭代次数 也即是block的数量 block的数量其实就是需要循环的维度的大小的乘积
    size_t total_blocks = 1;
    for (size_t i = 0; i < loop_dims; i++) {
        total_blocks *= src_shape[i];
    }

    // dst的偏移，由于dst连续，所以每拷贝一个block，偏移增加block_byte_size
    size_t dst_offset = 0;

    // 开始拷贝
    for (size_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // 计算源地址偏移，这里用elem偏移，减少乘法运算
        size_t src_offset_elems = 0;
        // 每个维度计算偏移
        for (size_t dim = 0; dim < loop_dims; dim++) {
            src_offset_elems += indices[dim] * src_strides[dim];
        }
        size_t src_offset_bytes = src_offset_elems * elem_size;

        // 执行拷贝
        api->memcpy_sync(
            dst_base_ptr + dst_offset,
            src_base_ptr + src_offset_bytes,
            block_byte_size,
            LLAISYS_MEMCPY_H2H);

        // 更新dst偏移
        dst_offset += block_byte_size;

        // 更新indices，从最后一个维度开始进位
        for (size_t dim = loop_dims - 1; dim >= 0; dim--) {
            // 由于拷贝完了这个block，当前维度索引加1
            indices[dim]++;
            // 检查是否需要进位
            if (indices[dim] < src_shape[dim]) {
                break;
            } 
            indices[dim] = 0; // 重置当前维度索引，又从最低维开始
        }
    }
}
} // namespace llaisys
