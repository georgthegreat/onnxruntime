#!/bin/sh

sed --in-place 's/s_library_cuda.Get()/GetProvider_CUDA()/g' "onnxruntime/core/session/provider_bridge_ort.cc"
