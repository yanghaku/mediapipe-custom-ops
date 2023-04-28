This a library to collect [MediaPipe] custom operators and apply these to [WasmEdge Wasi-NN plugin].

The source code in folder ```ops``` are copied from [MediaPipe].

The library impl the interface ```wasmEdgeWasiNnTfLiteAddCustomOps``` to be used in [WasmEdge Wasi-NN plugin].

```c++
extern "C" void wasmEdgeWasiNnTfLiteAddCustomOps(TfLiteInterpreterOptions *);
```

[MediaPipe]: https://github.com/google/mediapipe

[WasmEdge Wasi-NN plugin]: https://github.com/WasmEdge/WasmEdge
