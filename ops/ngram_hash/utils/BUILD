load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "ngram_hash_ops_utils",
    srcs = [
        "ngram_hash_ops_utils.cc",
    ],
    hdrs = [
        "ngram_hash_ops_utils.h",
    ],
    deps = [
        "//ops/ngram_hash/utils/utf",
    ],
    copts = COPTS,
    visibility = ["//ops/ngram_hash:__pkg__"],
)
