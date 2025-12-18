# MerkleDb NIF Bridge Report
**Date:** December 17, 2025
**Status:** ✅ Operational / Bridge Complete / ✅ Library Verified
**Bridge Version:** V5 (Auto-Allocation + Aligned Buffering)

---

## 1. Executive Summary

We have established a robust, automated NIF bridge connecting the `MerkleDb` Elixir application to the high-performance `FP_ASM_LIB_DEV` C/Assembly library.

**Key Achievements:**
*   **Total Access**: All 167 compatible functions in the core library headers are bridged.
*   **Safety**: The bridge (`gen_bridge.exs` V5) automatically handles memory safety:
    *   **Auto-Allocation**: For functions returning data via pointers, the NIF allocates binary resources automatically.
    *   **Aligned Buffering**: To prevent crashes with AVX2 instructions, the bridge allocates **32-byte aligned** temporary buffers for all pointer operations, copying data to/from Erlang binaries.
*   **Zero Boilerplate**: Users call `ASM.function(input_bin, output_size, ...)` and receive the result binary directly.
*   **Windows Stability**: Fixed critical ABI violation in the assembly library (stack alignment/unwinding) that was causing crashes.

**Current Status**:
*   The bridge logic is correct.
*   Integer operations (e.g., `fp_reduce_add_i64`) are **Working**.
*   Floating point operations (e.g., `fp_reduce_add_f64`) are **Working**.

---

## 2. Technical Architecture

### 2.1 The V5 Generator (`gen_bridge.exs`)
The generator is a sophisticated metaprogramming script that:
1.  **Whitelists Headers**: Scans only safe, non-conflicting headers (`fp_core.h`, `fp_stats.h`, etc.).
2.  **Parses C Signatures**: Understands `const*` (Input) vs `type*` (Output) semantics.
3.  **Generates Safe Wrappers**:
    *   Allocates aligned memory (via `_aligned_malloc`/`posix_memalign`).
    *   Copies input binary data to aligned memory.
    *   Calls the C function.
    *   Copies result from aligned memory to a new Erlang Binary.
    *   Frees temporary resources.
    *   Returns the binary (or tuple of binaries) to Elixir.

### 2.2 Compilation Chain
*   **ASM**: `nasm -f win64` (Windows x64).
*   **C**: `gcc -O3 -std=c11 -shared`.
*   **Link**: All object files linked into `merkle_nif.dll`.

---

## 3. Solved Issues

### 3.1 Floating Point Crash (Windows)
*   **Issue**: Calls to `fp_reduce_add_f64` caused immediate silent exit.
*   **Cause**: The vendored assembly library (`macros.inc`) used `and rsp, ...` to align the stack dynamically. On Windows x64, modifying `RSP` without emitting unwind information (.pdata) violates the ABI and breaks Exception Handling/Unwinding, causing the OS to terminate the process during context switches or exceptions. Additionally, `vmovdqa` (Aligned Store) requires 32-byte alignment which standard Windows stack (16-byte) does not guarantee without the illegal dynamic alignment.
*   **Resolution**: Patched `macros.inc`:
    1.  Removed dynamic alignment (`and rsp`).
    2.  Fixed stack allocation size to maintain 16-byte alignment (264 bytes).
    3.  Switched `vmovdqa` to `vmovdqu` (Unaligned Store) for saving non-volatile registers, making the code safe for the standard Windows stack.

### 3.2 "BadArg" on Output Size
*   **Issue**: Passing `size_t` arguments.
*   **Fix**: Implemented in V5 using `enif_get_uint64` which safely maps 64-bit integers from Elixir to C.

---

## 4. Usage Guide

### 4.1 Input/Output Pattern
For functions that modify an array in-place in C:
```c
void fp_map_scale_f64(const double* in, double* out, size_t n, double c);
```
**Elixir Call:**
```elixir
# Allocates 'out_size' bytes, computes, returns 'result_binary'
result_binary = MerkleDb.ASM.fp_map_scale_f64(input_binary, out_size, n, c)
```

For pure scalar functions:
```c
int64_t fp_reduce_add_i64(const int64_t* in, size_t n);
```
**Elixir Call:**
```elixir
sum = MerkleDb.ASM.fp_reduce_add_i64(input_binary, n)
```

### 4.2 Running Tests
```powershell
mix run example_usage.exs
```

---

## 6. Algorithms Status

The `FP_ASM_LIB_DEV` includes high-level algorithms (K-Means, PCA, Naive Bayes, Neural Networks).

*   **Supported**: Stateless utility functions (e.g., `fp_pca_generate_ellipse_data`) are bridged and working.
*   **Unsupported**: Core training/inference functions (e.g., `fp_kmeans_fit`, `fp_neural_network_train`) are **NOT** yet bridged.
    *   **Reason**: These functions return or manipulate complex C structs (`KMeansResult`, `NeuralNetwork`). The current V5 bridge handles arrays and scalars but does not yet support **Resource Objects** (wrapping C structs as Erlang resources).
    *   **Future Work**: A "V6" bridge is needed to map these C structs to opaque Erlang resources, allowing Elixir to hold a reference to a trained model and pass it back for prediction.

## 7. Conclusion

The `MerkleDb.ASM` module now provides a direct, high-performance, and **stable** link to the core math and statistical capabilities of the underlying assembly library on Windows.
