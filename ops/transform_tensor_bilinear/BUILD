load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "//ops/transform_tensor_bilinear/tf_patch/common:transform_tensor_bilinear",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:compatibility",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    copts = COPTS,
    visibility = ["//lib:__pkg__"],
)
