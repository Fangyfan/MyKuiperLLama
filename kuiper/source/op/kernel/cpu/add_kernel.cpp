#include <armadillo>
#include "add_kernel.h"

namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream) {
    UNUSED(stream);

    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK_EQ(input1.size(), output.size());
    CHECK_EQ(input2.size(), output.size());

    // arma::fvec 表示 单精度浮点向量（float vector），封装了连续的浮点数组，支持高效的逐元素运算
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 向量，避免内存拷贝的性能损耗
    // Col(float *aux_mem, const arma::uword aux_length, const bool copy_aux_mem, const bool strict)
    // (1) aux_mem: 数据指针：去掉 const（因为 Armadillo 向量需要非 const 指针）
    // (2) aux_length: 向量长度，张量的总元素数
    // (3) copy_aux_mem: false => 不拷贝数据（直接复用原始内存）
    // (4) strict: true => 严格模式（不允许 Armadillo 重新分配内存）
    arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
    output_vec = input1_vec + input2_vec;
}
}  // namespace kernel