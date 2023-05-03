COPTS = [
    "-std=c++14",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-fPIC",
    "-O3",
    "-fvisibility=hidden",
    "-fno-exceptions",
    "-Wl,--strip-all",
    "-Wno-sign-compare"
]
