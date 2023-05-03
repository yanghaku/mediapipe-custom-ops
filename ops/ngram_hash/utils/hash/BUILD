load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "murmur",
    srcs = ["murmur.cc"],
    hdrs = ["murmur.h", "integral_types.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
    ],
    copts = COPTS,
    visibility = ["//ops/ngram_hash:__pkg__"],
)
