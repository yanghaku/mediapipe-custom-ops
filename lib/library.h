#ifndef CUSTOM_OPS_LIBRARY_H
#define CUSTOM_OPS_LIBRARY_H

#if defined(_WIN32)
#define LIB_EXPORT __declspec(dllexport)
#else
#define LIB_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif // __cplusplus

// tensorflow/lite/c/c_api.h
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
// tensorflow/lite/c/common.h
typedef struct TfLiteRegistration TfLiteRegistration;
// tensorflow/lite/c/c_api_experimental.h
void LIB_EXPORT TfLiteInterpreterOptionsAddCustomOp(TfLiteInterpreterOptions *options, const char *name,
                                                    const TfLiteRegistration *registration, int32_t min_version,
                                                    int32_t max_version);

void LIB_EXPORT wasmEdgeWasiNnTfLiteAddCustomOps(TfLiteInterpreterOptions *);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // CUSTOM_OPS_LIBRARY_H
