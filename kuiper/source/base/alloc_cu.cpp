#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::DeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) {
        return nullptr;
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, byte_size);
    CHECK_EQ(err, cudaSuccess);
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (ptr == nullptr) {
        return;
    }
    cudaFree(ptr);
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base