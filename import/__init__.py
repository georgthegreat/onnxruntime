import os

from devtools.yamaker.modules import GLOBAL, Linkable, Recursable, Switch, Words
from devtools.yamaker.project import CMakeNinjaNixProject
from devtools.yamaker import abseil
from devtools.yamaker import onnx


# onnxruntime/core/providers/shared_library/provider_bridge_provider.cc
# duplicates (proxies) most of the Provider api between host and provider parts.
# In order to link them statically we have to resolve these duplicates somehow.
# Instead of applying huge patch to provider_bridge_provider.cc
# we remove it from SRCS during post_install and replace it
# with smaller parts extracted to srcs kept untouched by the means of keep_paths option.
ORT_CUDA_SRCS = {
    "core/providers/shared_library/provider_bridge_cpu.cc",
    "core/providers/shared_library/provider_bridge_einsum.cc",
    "core/providers/shared_library/provider_bridge_unload.cc",
}


def post_build(self):
    onnx.sanitize_proto_names(self)
    onnx.apply_tstring_replacements(self)


def post_install(self):
    with self.yamakes["onnxruntime/core/providers/cuda"] as ort_cuda:
        ort_cuda.SRCS.remove("core/providers/shared_library/provider_bridge_provider.cc")
        ort_cuda.SRCS |= ORT_CUDA_SRCS

        # Mark cuda_provider_factory GLOBAL to make weak symbol work
        # See:
        # https://stackoverflow.com/questions/13089166
        ort_cuda.SRCS.remove("core/providers/cuda/cuda_provider_factory.cc")
        ort_cuda.SRCS.add(GLOBAL("core/providers/cuda/cuda_provider_factory.cc"))

        # FIXME: provides.py does not work with cudatoolkit libraries at the time
        ort_cuda.PEERDIR += [
            "contrib/libs/nvidia/cub",
            "contrib/libs/nvidia/cublas",
            "contrib/libs/nvidia/cudnn",
            "contrib/libs/nvidia/cufft",
            "contrib/libs/nvidia/thrust",
        ]
        ort_cuda.ADDINCL += [
            "contrib/libs/nvidia/cudnn",
        ]

        # FIXME: unbundle_from does not work with this library at the time
        ort_cuda.PEERDIR += [
            "contrib/libs/onnx",
        ]
        ort_cuda.ADDINCL.remove(f"{self.arcdir}/_deps/abseil_cpp-src")

    with self.yamakes["."] as ort:
        ort.SRCS.remove("onnxruntime/core/providers/shared/common.cc")
        os.remove(f"{self.dstdir}/onnxruntime/core/providers/shared/common.cc")

        ort.after(
            "SRCS",
            Switch(
                OS_DARWIN=Linkable(
                    SRCS=["onnxruntime/core/platform/apple/logging/apple_log_sink.mm"],
                    EXTRALIBS=[Words("-framework Foundation")],
                ),
            ),
        )
        # Due to the static linkage against abseil-cpp,
        # we are unable to generate PEERDIRs properly,
        # though we unbundle it by the means of abseil.apply_unbundling_hacks() below.
        #
        # Fix dependency manually for the time being.
        ort.ADDINCL.remove(f"{self.arcdir}/_deps/abseil_cpp-src")
        ort.PEERDIR.add("contrib/restricted/abseil-cpp")

        # Disribute instrinsic-dependent sources between SRC_C_{OPT} macros.
        for opt in ["avx", "avx2", "avx512", "amx"]:
            for src in sorted(ort.SRCS):
                if f"/{opt}/" in src or src.endswith(f"_{opt}.cpp"):
                    ort.SRCS.remove(src)
                    ort.after("SRCS", f"SRC_C_{opt.upper()}({src})")

        ort.after(
            "RECURSE",
            Switch(
                {
                    "HAVE_CUDA AND CUDA_VERSION VERSION_GE 11.4": Recursable(RECURSE=ort.RECURSE),
                }
            ),
        )
        ort.RECURSE = []


onnx_runtime = CMakeNinjaNixProject(
    nixattr="onnxruntime",
    arcdir="contrib/libs/onnx_runtime",
    owners=["g:cpp-contrib", "g:matrixnet"],
    ignore_targets=[
        "flatc",
        # stands for protoc binary
        "protoc-3.20.2.0",
        # stands for protoc library
        "protoc",
        # Our version of clog (bundled cpuinfo dependency) is replaced by no-op.
        # See yamaker/projects/cpuinfo/files/include/clog.h for the details.
        # While we will unbudle cpuinfo by the means of unbundle_from from below,
        # clog dependency can be simply ignored.
        "clog",
    ],
    ignore_commands=[
        "protoc-3.20.2.0",
    ],
    unbundle_from={
        "boost_mp11": "_deps/mp11-src",
        "cpuinfo": "_deps/pytorch_cpuinfo-src",
        "eigen": "_deps/eigen-src",
        "flatbuffers": "_deps/flatbuffers-src",
        "nsync": "_deps/google_nsync-src",
        "nsync_cpp": "_deps/google_nsync-build",
        "onnx": "_deps/onnx-src",
        "onnx_proto": "_deps/onnx-build",
        "protobuf": "_deps/protobuf-src",
        "protobuf-lite": "_deps/protobuf-src",
        "re2": "_deps/re2-src",
        "nlohmann_json": "_deps/nlohmann_json-src",
    },
    copy_sources=[
        "onnxruntime/core/platform/apple/logging/apple_log_sink.h",
        "onnxruntime/core/platform/apple/logging/apple_log_sink.mm",
    ],
    # fmt: off
    keep_paths=[
        f"onnxruntime/{src}"
        for src in ORT_CUDA_SRCS
    ],
    # fmt: on
    put={
        "onnx": "cmake/external/onnx",
        "onnx_proto": "external/onnx",
        "onnxruntime_common": ".",
        "onnxruntime_providers_cuda": "onnxruntime/core/providers/cuda",
    },
    put_with={
        "onnxruntime_common": [
            "onnxruntime_flatbuffers",
            "onnxruntime_framework",
            "onnxruntime_graph",
            "onnxruntime_mlas",
            "onnxruntime_optimizer",
            "onnxruntime_providers",
            "onnxruntime_providers_shared",
            "onnxruntime_session",
            "onnxruntime_util",
        ],
    },
    disable_includes=[
        # if defined(__ANDROID__)
        "core/platform/android/logging/android_log_sink.h",
        # ifdef CONCURRENCY_VISUALIZER
        "cvmarkersobj.h",
        # ifdef USE_MIMALLOC
        "mimalloc.h",
        # #ifdef ORT_USE_NCCL
        "nccl.h",
        # ifdef USE_AZURE
        "core/framework/cloud_executor.h",
        "core/framework/cloud_invoker.h",
        "http_client.h",
        # if defined(__wasm__)
        "emscripten/*.h",
        "wasm_simd128.h",
        # ifdef ENABLE_ATEN
        "contrib_ops/cpu/aten_ops/aten_op_executor.h",
        # ifdef ENABLE_TRAINING
        "orttraining/**/*.h",
        "core/framework/orttraining_partial_executor.h",
        "core/framework/partial_graph_execution_state.h",
        "core/framework/program_region.h",
        # ifdef ENABLE_TRAINING_CORE
        "core/optimizer/compute_optimizer/compute_optimizer.h",
        "core/optimizer/compute_optimizer/passthrough_actors.h",
        # ifdef USE_DML
        "core/graph/dml_ops/dml_defs.h",
        # if defined DEBUG_NODE_INPUTS_OUTPUTS
        "core/common/path_utils.h",
        "core/framework/debug_node_inputs_outputs_utils.h",
        "sqlite3.h",
        # ifdef ENABLE_LANGUAGE_INTEROP_OPS
        "core/language_interop_ops/language_interop_ops.h",
        # ifndef DISABLE_CONTRIB_OPS
        "contrib_ops/cpu/aten_ops/aten_op.h",
        # ifdef ENABLE_EXTENSION_CUSTOM_OPS
        "onnxruntime_extensions.h",
        # ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
        "core/platform/tracing.h",
        "TraceLoggingActivity.h",
        # Various computational providers, might be enabled upon request
        "core/providers/acl/*.h",
        "core/providers/azure/azure_provider_factory_creator.h",
        "core/providers/cann/cann_execution_provider_info.h",
        "core/providers/coreml/*.h",
        "core/providers/armnn/*.h",
        "core/providers/dml/*.h",
        "core/providers/nnapi/*.h",
        "core/providers/nuphar/*.h",
        "core/providers/rknpu/*.h",
        "core/providers/rocm/cu_inc/common.cuh",
        "core/providers/snpe/*.h",
        "core/providers/tvm/*.h",
        "core/providers/vitisai/*.h",
        "core/providers/xnnpack/*.h",
        # Put under #if 0 by XXX.patch
        "core/providers/cuda/test/all_tests.h",
        # if USE_FLASH_ATTENTION
        "contrib_ops/cuda/bert/cutlass_fmha/fmha_launch_template.h",
    ],
    post_build=post_build,
    post_install=post_install,
)

onnx_runtime.copy_top_sources_except.add("ORT_icon_for_light_bg.png")


abseil.apply_unbundling_hacks(
    onnx_runtime,
    abseil_subdir="_deps/abseil_cpp-src",
    libs={
        "absl_bad_optional_access",
        "absl_bad_variant_access",
        "absl_base",
        "absl_city",
        "absl_civil_time",
        "absl_cord",
        "absl_cord_internal",
        "absl_cordz_functions",
        "absl_cordz_handle",
        "absl_cordz_info",
        "absl_debugging_internal",
        "absl_demangle_internal",
        "absl_exponential_biased",
        "absl_graphcycles_internal",
        "absl_hash",
        "absl_hashtablez_sampler",
        "absl_int128",
        "absl_log_severity",
        "absl_low_level_hash",
        "absl_malloc_internal",
        "absl_raw_hash_set",
        "absl_raw_logging_internal",
        "absl_spinlock_wait",
        "absl_stacktrace",
        "absl_strings",
        "absl_strings_internal",
        "absl_symbolize",
        "absl_synchronization",
        "absl_throw_delegate",
        "absl_time",
        "absl_time_zone",
    },
)
