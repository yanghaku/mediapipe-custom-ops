load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
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
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
    copts = COPTS,
    visibility = ["//ops/landmarks_to_transform_matrix:__pkg__"],
)
