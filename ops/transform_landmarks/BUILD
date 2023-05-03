load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "//ops/transform_landmarks/tf_patch/common:transform_landmarks",
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
