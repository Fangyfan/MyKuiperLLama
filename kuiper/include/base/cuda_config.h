#ifndef KUIPER_INCLUDE_BASE_CUDA_CONFIG_H
#define KUIPER_INCLUDE_BASE_CUDA_CONFIG_H

// cuBLAS 库（CUDA 加速的线性代数库，矩阵/向量运算）
// CUDA 运行时 API（核心 CUDA 操作，如流、内存管理）
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace kernel {

// 定义结构体 CudaConfig，封装 CUDA 流配置
// RAII 思想（Resource Acquisition Is Initialization），翻译为「资源获取即初始化」
// 资源绑定到对象生命周期：创建对象时获取资源，销毁对象时自动释放资源
// 避免手动管理：忘记释放（内存泄漏）、重复释放（运行时错误）、悬垂指针（提前释放）
// NoCopyable
struct CudaConfig {
    // 成员变量：CUDA 流句柄（指向 CUDA 流的指针），默认初始化为 nullptr
    // CUDA 流是 GPU 操作的「任务队列」—— 默认情况下，GPU 操作都在「默认流（nullptr）」中串行执行；
    // 自定义流可以让多个 GPU 任务并行执行（比如一个流做矩阵乘法，另一个流做数据拷贝）
    cudaStream_t stream = nullptr;

    // 禁用拷贝构造
    CudaConfig(const CudaConfig&) = delete;

    // 禁用赋值运算符
    CudaConfig& operator=(const CudaConfig&) = delete;

    // 允许移动构造（可选，避免资源浪费）
    CudaConfig(CudaConfig&& other) noexcept {
        stream = other.stream;
        other.stream = nullptr;  // 转移所有权，避免原对象析构时销毁
    }

    // 析构函数：自动销毁 CUDA 流，释放资源
    ~CudaConfig() {
        if (stream) {
            cudaError_t err = cudaStreamDestroy(stream);
            CHECK_EQ(err, cudaSuccess);
        }
    }
};

}  // namespace kernel

#endif  // KUIPER_INCLUDE_BASE_CUDA_CONFIG_H