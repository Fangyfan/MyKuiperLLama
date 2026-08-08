#include <algorithm>
#include <cuda_runtime.h>
#include "sampler/argmax_sampler.h"
#include "../op/kernel/cuda/argmax_kernel.cuh"

namespace sampler {
ArgmaxSampler::ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

int32_t ArgmaxSampler::sample(
    const float* logits, 
    int32_t size, 
    int32_t* argmax_token, 
    void* argmax_buffer, 
    int32_t* output_cpu, 
    void* stream
) const {
    sample_async(logits, size, argmax_token, argmax_buffer, output_cpu, stream);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        // 同步保证 D2H 拷贝完成后再读取 pinned memory 上的结果
        if (stream) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        } else {
            cudaStreamSynchronize(nullptr);
        }
    }
    return *output_cpu;
}

void ArgmaxSampler::sample_async(
    const float* logits, 
    int32_t size, 
    int32_t* argmax_token, 
    void* argmax_buffer, 
    int32_t* output_cpu, 
    void* stream
) const {
    CHECK_NE(logits, nullptr);
    CHECK_GT(size, 0);
    CHECK_NE(output_cpu, nullptr);
    if (device_type_ == base::DeviceType::DeviceCPU) {
        *output_cpu = std::distance(logits, std::max_element(logits, logits + size));
    } else if (device_type_ == base::DeviceType::DeviceCUDA) {
        kernel::argmax_kernel_cu(logits, size, argmax_token, argmax_buffer, output_cpu, stream);
    } else {
        LOG(FATAL) << "Unknown device type for argmax sampler." << std::endl;
    }
}
}  // namespace sampler
