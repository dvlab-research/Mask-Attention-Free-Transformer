#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "attention/attention_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "attention_forward");
    m.def("attention_backward", &attention_backward, "attention_backward");
}
