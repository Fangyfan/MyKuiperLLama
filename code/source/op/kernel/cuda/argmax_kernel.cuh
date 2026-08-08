#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cstdint>

namespace kernel {
// argmax 结果通过 cudaMemcpyAsync 写入 output_cpu（必须是 pinned memory），调用方负责 stream 同步后再读取
void argmax_kernel_cu(
    const float* input, 
    int32_t size, /* 151936 */
    int32_t* argmax_token, 
    void* argmax_buffer, 
    int32_t* output_cpu, 
    void* stream
);

// device 上的 pos 自增 1 (CUDA Graph 图尾节点)
void pos_increment_kernel_cu(int32_t* pos, void* stream);

}  // namespace kernel

#endif  // ARGMAX_KERNEL_CUH
