load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "ngram_hash",
    srcs = ["ngram_hash.cc"],
    hdrs = ["ngram_hash.h"],
    deps = [
        "//ops/ngram_hash/utils:ngram_hash_ops_utils",
        "//ops/ngram_hash/utils/hash:murmur",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
    copts = COPTS,
    visibility = ["//lib:__pkg__"],
    alwayslink = 1,
)
