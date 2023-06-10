// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the provider DLL side of the provider API to let providers be built as a DLL

#include "provider_api.h"
#include "core/providers/shared/common.h"

namespace onnxruntime {

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// "Global initializer calls a non-constexpr function."
#pragma warning(disable : 26426)
#endif
ProviderHostCPU& g_host_cpu = Provider_GetHost()->GetProviderHostCPU();
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

}  // namespace onnxruntime
