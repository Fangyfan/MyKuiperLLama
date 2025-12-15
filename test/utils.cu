#include <glog/logging.h>
#include "utils.cuh"

__global__ void test_function_cu(float* arr_cu, int32_t size, float value) {
    // 计算全局线程索引
    int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    // 边界检查：确保不越界
    if (tid >= size) {
        return;
    }
    
    // 每个线程设置一个元素
    arr_cu[tid] = value;
}

// 将 GPU 数组的所有元素设置为指定的 value
// arr_cu: 指向 GPU 内存的浮点数组指针
// size: 数组的元素个数
// value: 要设置的值
void set_value_cu(float* arr_cu, int32_t size, float value) {
    // 每个线程块的线程数 512，线程块的大小通常是 32 的倍数（warp 大小）
    int32_t thread_num = 512;
    
    // 线程块数量，确保有足够的线程覆盖所有数组元素
    int32_t block_num = (size + thread_num - 1) / thread_num;
    
    // 等待之前所有 CUDA 操作完成，确保不会与即将启动的核函数重叠执行
    cudaDeviceSynchronize();

    // 获取最后一个 CUDA 错误，CUDA 调用是异步的，错误可能延迟报告，提前清理错误状态，避免混淆错误来源
    const cudaError_t err2 = cudaGetLastError();

    // 启动 CUDA 核函数 (线程网格中的线程块数量 × 每个线程块的线程数)
    test_function_cu<<<block_num, thread_num>>>(arr_cu, size, value);

    // 等待核函数执行完成，并检查核函数错误
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    CHECK_EQ(err, cudaSuccess);
}