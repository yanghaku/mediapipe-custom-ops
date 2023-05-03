workspace(name = "mediapipe_custom_ops")

TF_SRC="tensorflow_src"
TFLITE_C_LIB="tflite_c_lib"

local_repository(
    name = "org_tensorflow",
    path = TF_SRC,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

new_local_repository(
    name = "tflite_c",
    path = TFLITE_C_LIB,
    build_file_content = "cc_import(name = \"tflite_c\", shared_library=\"libtensorflowlite_c.so\", visibility = [\"//visibility:public\"])",
)
