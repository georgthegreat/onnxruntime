#!/bin/sh

sed --in-place 's/\xEF\xBB\xBF//' onnxruntime/contrib_ops/cuda/math/fft_ops_impl.cu
