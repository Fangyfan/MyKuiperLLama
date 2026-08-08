#ifndef CODE_INCLUDE_SAMPLER_H
#define CODE_INCLUDE_SAMPLER_H

#include "base/base.h"

namespace sampler {
class Sampler {
public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

    // argmax 贪心采样: 模型输出的概率分布 logits，其长度为 size，输出概率最高的那个 token 的索引
    // 输入 logits = [-0.2, 2.3, 0.5, 1.8, -1.0]，输出为索引 1
    // output_cpu: CUDA 模式下必须是 pinned memory，采样结果必须经异步 D2H 拷贝写入
    virtual int32_t sample(const float* logits, int32_t size, int32_t* argmax_token, void* argmax_buffer, int32_t* output_cpu, void* stream = nullptr) const = 0;

    // 与 sample 相同，但不做流同步（结果尚未写回 output_cpu），用于 CUDA Graph capture
    virtual void sample_async(const float* logits, int32_t size, int32_t* argmax_token, void* argmax_buffer, int32_t* output_cpu, void* stream = nullptr) const = 0;

protected:
    base::DeviceType device_type_ = base::DeviceType::DeviceUnknown;
};
}  // namespace sampler

#endif  // CODE_INCLUDE_SAMPLER_H
