load("//ops:variables.bzl", "COPTS")

cc_library(
    name = "utf",
    srcs = [
        "rune.c",
        "runetype.c",
        "runetypebody.h",
    ],
    hdrs = ["utf.h"],
    copts = COPTS,
    visibility = [
        "//ops/ngram_hash:__pkg__",
        "//ops/ngram_hash/utils:__pkg__",
    ],
)
