all:
	BAZEL_LINKLIBS=-l%:libstdc++.a bazel build -c opt --experimental_repo_remote_exec --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //lib:mediapipe_custom_ops --sandbox_debug --verbose_failures

clean:
	bazel clean

.phony: all config clean
