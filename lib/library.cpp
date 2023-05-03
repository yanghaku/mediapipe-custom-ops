#include "library.h"

// ops
#include "ops/kmeans_embedding_lookup/kmeans_embedding_lookup.h"
#include "ops/landmarks_to_transform_matrix/landmarks_to_transform_matrix.h"
#include "ops/max_pool_argmax/max_pool_argmax.h"
#include "ops/max_unpooling/max_unpooling.h"
#include "ops/ngram_hash/ngram_hash.h"
#include "ops/ragged/ragged_tensor_to_tensor_tflite.h"
#include "ops/sentencepiece/sentencepiece_tokenizer_tflite.h"
#include "ops/transform_landmarks/transform_landmarks.h"
#include "ops/transform_tensor_bilinear/transform_tensor_bilinear.h"
#include "ops/transpose_conv_bias/transpose_conv_bias.h"

void wasmEdgeWasiNnTfLiteAddCustomOps(TfLiteInterpreterOptions *options) {
    TfLiteInterpreterOptionsAddCustomOp(options, "MaxPoolingWithArgmax2D",
                                        mediapipe::tflite_operations::RegisterMaxPoolingWithArgmax2D(), 1, 1);

    TfLiteInterpreterOptionsAddCustomOp(options, "MaxUnpooling2D",
                                        mediapipe::tflite_operations::RegisterMaxUnpooling2D(), 1, 1);

    TfLiteInterpreterOptionsAddCustomOp(options, "Convolution2DTransposeBias",
                                        mediapipe::tflite_operations::RegisterConvolution2DTransposeBias(), 1, 1);

    TfLiteInterpreterOptionsAddCustomOp(options, "TransformTensor",
                                        mediapipe::tflite_operations::RegisterTransformTensorBilinearV1(), 1, 1);
    TfLiteInterpreterOptionsAddCustomOp(options, "TransformTensorBilinear",
                                        mediapipe::tflite_operations::RegisterTransformTensorBilinearV2(), 2, 2);

    TfLiteInterpreterOptionsAddCustomOp(options, "TransformLandmarks",
                                        mediapipe::tflite_operations::RegisterTransformLandmarksV1(), 1, 1);
    TfLiteInterpreterOptionsAddCustomOp(options, "TransformLandmarks",
                                        mediapipe::tflite_operations::RegisterTransformLandmarksV2(), 2, 2);

    TfLiteInterpreterOptionsAddCustomOp(options, "Landmarks2TransformMatrix",
                                        mediapipe::tflite_operations::RegisterLandmarksToTransformMatrixV1(), 1, 1);
    TfLiteInterpreterOptionsAddCustomOp(options, "Landmarks2TransformMatrix",
                                        mediapipe::tflite_operations::RegisterLandmarksToTransformMatrixV2(), 2, 2);

    TfLiteInterpreterOptionsAddCustomOp(options, "NGramHash", mediapipe::tflite_operations::Register_NGRAM_HASH(), 1,
                                        1);
    TfLiteInterpreterOptionsAddCustomOp(options, "KmeansEmbeddingLookup",
                                        mediapipe::tflite_operations::Register_KmeansEmbeddingLookup(), 1, 1);
    TfLiteInterpreterOptionsAddCustomOp(options, "TFSentencepieceTokenizeOp",
                                        mediapipe::tflite_operations::Register_SENTENCEPIECE_TOKENIZER(), 1, 1);
    TfLiteInterpreterOptionsAddCustomOp(options, "RaggedTensorToTensor",
                                        mediapipe::tflite_operations::Register_RAGGED_TENSOR_TO_TENSOR(), 1, 1);
}
