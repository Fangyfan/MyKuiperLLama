#include "argmax_kernel.cuh"
#include "cuda_runtime.h"
#include "base/buffer.h"
#include <cfloat>

namespace kernel {
template <int32_t WARP_SIZE>
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, WARP_SIZE);

        if (other_idx != -1) {
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) {
                idx = other_idx;
            }
        }
    }
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx) {
    __shared__ float shared_vals[WARP_NUM];
    __shared__ int32_t shared_idxs[WARP_NUM];

    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;

    warp_reduce_argmax<32>(val, idx);
    if (lane == 0) {
        shared_vals[warp] = val;
        shared_idxs[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
            idx = shared_idxs[lane];
        }
        warp_reduce_argmax<WARP_NUM>(val, idx);
    }
}

struct __align__(8) val_idx {
    float val;
    int32_t idx;
};

template <int32_t SM_NUM, int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void argmax_kernel_1(
    const float* __restrict__ input,
    int32_t size,
    val_idx* __restrict__ temp
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    int32_t size4 = size >> 2;
    const float4* __restrict__ input4 = reinterpret_cast<const float4*>(input);

    float val = -INFINITY;
    int32_t idx = -1;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size4; i += SM_NUM * BLOCK_DIM) {
        float4 other_val4 = input4[i];
        if (val < other_val4.x) { val = other_val4.x; idx = (i << 2); }
        if (val < other_val4.y) { val = other_val4.y; idx = (i << 2) + 1; }
        if (val < other_val4.z) { val = other_val4.z; idx = (i << 2) + 2; }
        if (val < other_val4.w) { val = other_val4.w; idx = (i << 2) + 3; }
    }
    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        temp[blockIdx.x] = {val, idx};
    }
}

static __global__ void argmax_kernel_2(
    const val_idx* __restrict__ temp,
    int32_t* __restrict__ output
) {
    float val = -INFINITY;
    int32_t idx = -1;
    int32_t tid = threadIdx.x << 2;

#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        val_idx other = temp[tid + i];
        if (val < other.val) {
            val = other.val;
            idx = other.idx;
        } else if (val == other.val && idx > other.idx) {
            idx = other.idx;
        }
    }
    warp_reduce_argmax<32>(val, idx);

    if (threadIdx.x == 0) {
        *output = idx;
    }
}

void argmax_kernel_cu(
    const float* input, 
    int32_t size, /* 151936 */
    int32_t* argmax_token, 
    void* argmax_buffer, 
    int32_t* output_cpu, 
    void* stream
) {
    int32_t* output_idx_cu_ptr = argmax_token;
    val_idx* temp = reinterpret_cast<val_idx*>(argmax_buffer);

    constexpr int32_t SM_NUM = 128;
    constexpr int32_t BLOCK_DIM = 512;
    // dim3 gridDim(SM_NUM);
    // dim3 blockDim(BLOCK_DIM);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    argmax_kernel_1<SM_NUM, BLOCK_DIM><<<SM_NUM, BLOCK_DIM, 0, stream_>>>(input, size, temp);
    argmax_kernel_2<<<1, 32, 0, stream_>>>(temp, output_idx_cu_ptr);
    // 不做流同步: 同步由调用方负责，这样本函数可以被 CUDA Graph capture
    // output_cpu 必须是 pinned memory，保证 cudaMemcpyAsync 可以被 capture
    cudaMemcpyAsync(output_cpu, output_idx_cu_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
}

static __global__ void pos_increment_kernel(int32_t* pos) {
    pos[0] += 1;
}

void pos_increment_kernel_cu(int32_t* pos, void* stream) {
    // pos 自增放在图尾，供下一拍 replay 使用（decode 稳态下 host 不再 H2D pos）
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    pos_increment_kernel<<<1, 1, 0, stream_>>>(pos);
}
}  // namespace kernel