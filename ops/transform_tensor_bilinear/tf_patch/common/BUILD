load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model_builder_helper",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model_transformer",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:object_reader",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:operation_parser",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:status",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
    copts = COPTS,
    visibility = ["//ops/transform_tensor_bilinear:__pkg__"],
)
