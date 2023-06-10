// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built as a DLL

#include "provider_api.h"
#include "core/providers/cpu/math/einsum.h"

namespace onnxruntime {

template <>
std::unique_ptr<EinsumTypedComputeProcessor<float>> EinsumTypedComputeProcessor<float>::Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) { return Provider_GetHost()->GetProviderHostCPU().EinsumTypedComputeProcessor_float__Create(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
template <>
std::unique_ptr<EinsumTypedComputeProcessor<double>> EinsumTypedComputeProcessor<double>::Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) { return Provider_GetHost()->GetProviderHostCPU().EinsumTypedComputeProcessor_double__Create(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }
template <>
std::unique_ptr<EinsumTypedComputeProcessor<MLFloat16>> EinsumTypedComputeProcessor<MLFloat16>::Create(OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets) { return Provider_GetHost()->GetProviderHostCPU().EinsumTypedComputeProcessor_MLFloat16__Create(context, allocator, tp, einsum_compute_preprocessor, einsum_cuda_assets); }

}  // namespace onnxruntime
