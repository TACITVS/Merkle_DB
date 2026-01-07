# Module 1: Getting Started

Welcome to the MerkleDb Developer Course! In this module, we will set up your development environment and build the database from source.

## 1. Prerequisites

MerkleDb requires specific tools to compile the Assembly kernels and the Elixir bridge.

### Windows (Recommended)
- **Elixir**: Install via [installer](https://elixir-lang.org/install.html#windows).
- **MSYS2 / MinGW64**: Required for `gcc` and `make`.
- **NASM**: Install via [nasm.us](https://www.nasm.us/) and add to your PATH.

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install elixir nasm build-essential
```

---

## 2. Cloning the Repository

```bash
git clone https://github.com/TACITVS/Merkle_DB.git
cd Merkle_DB
```

---

## 3. Initializing the Project

MerkleDb uses a custom **NIF Bridge Generator**. Before compiling the Elixir code, you must generate the C glue code that connects Elixir to the Assembly kernels.

### Step 3.1: Fetch Dependencies
```bash
mix deps.get
```

### Step 3.2: Generate the Bridge
This script analyzes the native signatures and generates `native/generated_nif.c`.
```bash
mix run gen_bridge.exs
```

### Step 3.3: Compile
This will trigger the `Makefile` to compile the Assembly kernels and the C bridge, followed by the Elixir code.
```bash
mix compile
```

---

## 4. Running your first instance

Start an interactive Elixir shell (`iex`) with the database running:

```bash
iex -S mix
```

If everything is correct, you should see the MerkleDb banner:

```text
╔════════════════════════════════════════════════════════════╗
║  MerkleDB Vector Database - Ready                          ║
║  AVX2-Accelerated • Zero-Copy NIFs • IVF Indexing         ║
║  Server: http://localhost:4000                             ║
╚════════════════════════════════════════════════════════════╝
```

**Congratulations!** You have successfully built MerkleDb. In the next module, we will explore core operations.
