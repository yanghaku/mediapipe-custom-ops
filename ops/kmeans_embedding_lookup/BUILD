load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "kmeans_embedding_lookup",
    srcs = ["kmeans_embedding_lookup.cc"],
    hdrs = ["kmeans_embedding_lookup.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
    copts = COPTS,
    visibility = ["//lib:__pkg__"],
)
