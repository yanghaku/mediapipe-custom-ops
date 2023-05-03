load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "mediapipe_custom_ops_internal",
    srcs = ["library.cpp"],
    hdrs = ["library.h"],
    copts = COPTS,
    deps = [
        "//ops/max_pool_argmax:max_pool_argmax",
        "//ops/max_unpooling:max_unpooling",
        "//ops/transpose_conv_bias:transpose_conv_bias",
        "//ops/transform_tensor_bilinear:transform_tensor_bilinear",
        "//ops/transform_landmarks:transform_landmarks",
        "//ops/landmarks_to_transform_matrix:landmarks_to_transform_matrix",
        "//ops/ngram_hash:ngram_hash",
        "//ops/kmeans_embedding_lookup:kmeans_embedding_lookup",
        "//ops/sentencepiece:sentencepiece_tokenizer_tflite",
        "//ops/ragged:ragged_tensor_to_tensor_tflite",
    ],
    visibility = ["//lib:__pkg__"],
    alwayslink = 1,
)

cc_binary(
    name = "mediapipe_custom_ops",
    copts = COPTS,
    deps = [
        "mediapipe_custom_ops_internal",
        "@tflite_c//:tflite_c",
    ],
    linkshared = True,
)
