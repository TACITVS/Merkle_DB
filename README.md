# MerkleDb

A high-performance database based on Merkle Trees and DAGs, utilizing a specialized C/Assembly library (`FP_ASM_LIB_DEV`) for vector operations.

## Architecture

This project integrates a high-performance C and Assembly library into Elixir using NIFs (Native Implemented Functions).

*   **Elixir**: Core application logic, database management.
*   **C/Assembly**: Heavy lifting for vector math, statistics, and linear regression via `MerkleDb.ASM`.
*   **Bridge**: A custom generator (`gen_bridge.exs`) automatically parses C headers and creates safe, memory-aligned NIF wrappers.

## Setup & Compilation

**Requirements:**
*   Elixir ~> 1.14
*   GCC (MinGW64 on Windows)
*   NASM (Netwide Assembler)
*   Make (mingw32-make)

**Build:**
The project uses a custom `Makefile` integrated with `elixir_make`.

```powershell
mix deps.get
mix compile
```

This will:
1.  Compile the Assembly kernels (`.asm` -> `.obj`).
2.  Compile the C wrapper and algorithms.
3.  Link everything into `priv/merkle_nif.dll`.

## Usage

### Web Interface (Frontend)
The easiest way to use MerkleDB is via the built-in web dashboard.

1.  Run `start_server.bat` (or `mix run --no-halt`).
2.  Open your browser to `http://localhost:4000`.
3.  Click **"Initialize / Ingest Data"** to load the Bible corpus.
4.  Type a query (e.g., "creation", "light", "jesus") to search instantly.

### Elixir API
The bridge exposes the C library via the `MerkleDb.ASM` module.

```elixir
alias MerkleDb.ASM

# Scalar Reduction
input = <<1, 2, 3, 4, 5 :: little-integer-size(64)>>
count = 5
sum = ASM.fp_reduce_add_i64(input, count)
# sum == 15

# Vector Operation (Auto-Allocation)
# C Signature: void fp_map_scale_f64(const double* in, double* out, size_t n, double c)
# Elixir: Returns result binary
floats = <<1.0, 2.0 :: little-float-size(64)>>
result = ASM.fp_map_scale_f64(floats, 16, 2, 2.0)
```

See `example_usage.exs` for a runnable demo.

## Bridge Details

The NIF bridge is **auto-generated** by `gen_bridge.exs`.
*   **Version**: V5
*   **Features**: Header whitelisting, automatic binary allocation for output pointers, and **32-byte aligned memory buffering** to support AVX2 instructions safely.
*   **Report**: See `BRIDGE_REPORT.md` for a deep dive into the technical implementation.
*   **Upstream Fixes**: See `UPSTREAM_PATCH_GUIDE.md` for details on the critical Windows ABI fix applied to the assembly library.

## Known Issues

*   (None currently known. The Windows floating-point crash has been patched.)