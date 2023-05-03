load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//ops:variables.bzl", "COPTS")

package(default_visibility = ["//lib:__pkg__"])

filegroup(
    name = "config_fbs",
    srcs = ["config.fbs"],
)

flatbuffer_cc_library(
    name = "config",
    srcs = [
        "config.fbs",
    ],
)

flatbuffer_cc_library(
    name = "encoder_config",
    srcs = [
        "encoder_config.fbs",
    ],
    includes = [":config_fbs"],
)

cc_library(
    copts = COPTS,
    name = "utils",
    hdrs = [
        "utils.h",
    ],
)

cc_library(
    name = "double_array_trie",
    hdrs = [
        "double_array_trie.h",
    ],
    copts = COPTS,
    deps = [
        ":config",
        ":utils",
    ],
)

cc_library(
    name = "double_array_trie_builder",
    srcs = [
        "double_array_trie_builder.cc",
    ],
    hdrs = [
        "double_array_trie_builder.h",
    ],
    copts = COPTS,
    deps = ["@darts_clone"],
)

cc_library(
    name = "sentencepiece_constants",
    hdrs = ["sentencepiece_constants.h"],
    copts = COPTS,
)

cc_library(
    name = "model_converter",
    srcs = [
        "model_converter.cc",
    ],
    hdrs = [
        "model_converter.h",
    ],
    copts = COPTS,
    deps = [
        ":config",
        ":double_array_trie_builder",
        ":encoder_config",
        ":sentencepiece_constants",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_sentencepiece//src:sentencepiece_model_cc_proto",
    ],
)

cc_library(
    name = "optimized_encoder",
    srcs = [
        "optimized_encoder.cc",
    ],
    hdrs = [
        "optimized_encoder.h",
    ],
    copts = COPTS,
    deps = [
        ":double_array_trie",
        ":encoder_config",
        ":utils",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_tflite",
    srcs = ["sentencepiece_tokenizer_tflite.cc"],
    hdrs = ["sentencepiece_tokenizer_tflite.h"],
    copts = COPTS,
    deps = [
        ":optimized_encoder",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)
