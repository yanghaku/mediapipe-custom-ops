load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "//ops/landmarks_to_transform_matrix/tf_patch/common:landmarks_to_transform_matrix",
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
