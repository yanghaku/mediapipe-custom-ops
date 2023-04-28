cmake_minimum_required(VERSION 3.26)
project(mediapipe_custom_ops)

if (NOT DEFINED TENSORFLOW_SRC)
    message(ERROR "MUST SET TENSORFLOW_SRC")
else ()
    include_directories("${TENSORFLOW_SRC}")
endif ()

if (NOT DEFINED TENSORFLOW_LIB)
    message(ERROR "MUST SET TENSORFLOW_LIB")
else ()
    link_directories(${TENSORFLOW_LIB})
    link_libraries(tensorflowlite tensorflowlite_c)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fPIE -O3 -fno-rtti -w -Wextra -Werror -fvisibility=hidden")


file(GLOB_RECURSE OPS_FILES ops/**)
add_library(mediapipe_custom_ops SHARED library.cpp ${OPS_FILES})
