#include "library.h"
#include <tensorflow/lite/c/c_api_experimental.h>

// ops
#include "./ops/transpose_conv_bias/transpose_conv_bias.h"

[[maybe_unused]] void wasmEdgeWasiNnTfLiteAddCustomOps(TfLiteInterpreterOptions *options) {
    TfLiteInterpreterOptionsAddCustomOp(options, "Convolution2DTransposeBias",
                                        mediapipe::tflite_operations::RegisterConvolution2DTransposeBias(),
                                        1, 1);
}
