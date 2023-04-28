#ifndef CUSTOM_OPS_LIBRARY_H
#define CUSTOM_OPS_LIBRARY_H

#include <tensorflow/lite/c/c_api.h>

#if defined(_WIN32)
#define LIB_EXPORT __declspec(dllexport)
#else
#define LIB_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

[[maybe_unused]] void LIB_EXPORT wasmEdgeWasiNnTfLiteAddCustomOps(TfLiteInterpreterOptions *);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif //CUSTOM_OPS_LIBRARY_H
