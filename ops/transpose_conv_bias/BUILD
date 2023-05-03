load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "transpose_conv_bias",
    srcs = ["transpose_conv_bias.cc"],
    hdrs = ["transpose_conv_bias.h"],
    copts = [COPTS],
    visibility = ["//lib:__pkg__"],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)
