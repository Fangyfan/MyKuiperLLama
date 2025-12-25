#include "kernel_interface.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::DeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::DeviceCUDA) {
        return add_kernel_cu;
    }
    LOG(FATAL) << "Unknown device type for get a add kernel.\n";
    return nullptr;
}

}  // namespace kernel