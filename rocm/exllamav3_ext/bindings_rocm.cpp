// ROCm subset of exllamav3_ext: exactly the ops the vllm-exl3 plugin and LinearEXL3 use.
// Everything else in the upstream extension (attention, sampling, cache, quantizer, MoE
// tensor-core kernels, ...) is not built here. The Python package resolves ext.* lazily,
// so a missing op only fails if that path is taken.
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hgemm.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"
#include "quant/exl3_devctx.cuh"
#include "libtorch/linear.h"
#include "quant/exl3_moe.cuh"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.attr("EXL3_ROCM_PORT") = true;

    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("reconstruct_slice", &reconstruct_slice, "reconstruct_slice");
    m.def("reconstruct_had_slice", &reconstruct_had_slice, "reconstruct_had_slice");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("had_r_128_dual", &had_r_128_dual, "had_r_128_dual");
    m.def("hgemm", &hgemm, "hgemm");
    m.def("g_get_cc", &g_get_cc, "g_get_cc");
    m.def("g_get_num_sms", &g_get_num_sms, "g_get_num_sms");
    m.def("exl3_moe_max_concurrency", &exl3_moe_max_concurrency, "exl3_moe_max_concurrency");
    m.def("exl3_moe", &exl3_moe, "exl3_moe (ROCm batched implementation; accepts num_active)",
          py::arg("hidden_state"), py::arg("output_state"), py::arg("expert_count"), py::arg("token_sorted"),
          py::arg("weight_sorted"), py::arg("temp_state_g"), py::arg("temp_state_u"), py::arg("temp_intermediate_g"),
          py::arg("temp_intermediate_u"), py::arg("act_function"), py::arg("K_gate"), py::arg("K_up"), py::arg("K_down"),
          py::arg("gate_ptrs_trellis"), py::arg("gate_ptrs_suh"), py::arg("gate_ptrs_svh"),
          py::arg("up_ptrs_trellis"), py::arg("up_ptrs_suh"), py::arg("up_ptrs_svh"),
          py::arg("down_ptrs_trellis"), py::arg("down_ptrs_suh"), py::arg("down_ptrs_svh"),
          py::arg("gate_mcg"), py::arg("gate_mul1"), py::arg("up_mcg"), py::arg("up_mul1"),
          py::arg("down_mcg"), py::arg("down_mul1"), py::arg("act_limit"), py::arg("num_active") = -1);

    py::class_<BC_LinearFP16, std::shared_ptr<BC_LinearFP16>>(m, "BC_LinearFP16")
        .def(py::init<at::Tensor, c10::optional<at::Tensor>>(), py::arg("weight"), py::arg("bias") = py::none())
        .def("run", &BC_LinearFP16::run);

    py::class_<BC_LinearEXL3, std::shared_ptr<BC_LinearEXL3>>(m, "BC_LinearEXL3")
        .def(py::init<at::Tensor, at::Tensor, at::Tensor, int, c10::optional<at::Tensor>, bool, bool, at::Tensor>(),
             py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("K"), py::arg("bias"),
             py::arg("mcg"), py::arg("mul1"), py::arg("xh"))
        .def("run", &BC_LinearEXL3::run)
        .def("run_alloc", &BC_LinearEXL3::run_alloc);
}
