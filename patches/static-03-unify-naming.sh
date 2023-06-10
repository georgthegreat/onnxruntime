#!/bin/sh

find . -type f -exec sed --in-place 's/GetTypeFromOnnxType/TensorTypeFromONNXEnum/g' '{}' ';'
sed --in-place 's/ GetEnvironmentVar/ Env::Default().GetEnvironmentVar/g' "onnxruntime/core/platform/env_var_utils.h"
