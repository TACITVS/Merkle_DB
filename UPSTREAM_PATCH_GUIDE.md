# Upstream Library Patch Guide: Windows x64 ABI Fix

**Target Repository:** `FP_ASM_LIB_DEV` (Remote)
**Affected File:** `src/asm/macros.inc`
**Issue:** Windows Floating Point Crash / Stack Alignment Violation

---

## 1. The Issue

The original implementation of the `PROLOGUE` and `EPILOGUE` macros in `macros.inc` attempted to manually align the stack to 32 bytes (for AVX `vmovdqa` instructions) using dynamic stack pointer manipulation (`and rsp, ...`).

**Why this fails on Windows x64:**
1.  **ABI Violation:** The Windows x64 Application Binary Interface (ABI) relies on **Structured Exception Handling (SEH)**. Functions that allocate stack space must use static frame layouts or emit `.pdata` unwind information. Modifying `RSP` dynamically prevents the OS from unwinding the stack correctly during exceptions or context switches, leading to immediate process termination (silent crash).
2.  **Alignment Mismatch:** The Windows stack is guaranteed to be 16-byte aligned. The original code assumed it could realign to 32 bytes and restore the pointer later, but `vmovdqa` (Aligned Move) fails if the alignment logic conflicts with the OS's expectations or if the restore logic relies on a clobbered register/memory location during unwinding.

**Symptoms:**
*   Integer operations (e.g., `fp_reduce_add_i64`) often worked (likely due to lucky alignment or simpler register usage).
*   Floating-point operations (e.g., `fp_reduce_add_f64`, `fp_map_scale_f64`) caused an immediate **Access Violation (Segfault)** or silent exit when called from the NIF.

---

## 2. The Solution

The fix involves making the macros compliant with the Windows x64 ABI by:
1.  **Removing Dynamic Alignment:** Trust the OS's 16-byte alignment.
2.  **Using Unaligned Moves:** Switch from `vmovdqa` (Aligned) to `vmovdqu` (Unaligned) for saving/restoring XMM/YMM registers. Modern CPUs handle unaligned loads/stores with negligible penalty, and this is safe on a 16-byte aligned stack.
3.  **Fixed Stack Allocation:** Calculate the exact stack size needed to preserve alignment.
    *   Pushed registers: `RBP, RBX, R12, R13, R14, R15` (6 * 8 = 48 bytes).
    *   Return address: 8 bytes.
    *   Total before alloc: 56 bytes.
    *   Goal: 16-byte alignment. `56 + Alloc` must be a multiple of 16.
    *   `56 % 16 = 8`. So `Alloc` must be `8 (mod 16)`.
    *   Space needed for 8 YMM registers (32 bytes each): 256 bytes.
    *   `256` is divisible by 16.
    *   Add 8 bytes padding -> **264 bytes**. `(56 + 264) = 320` (divisible by 16).

---

## 3. Patch Diff (`src/asm/macros.inc`)

### PROLOGUE Macro

**Original (Broken on Windows):**
```nasm
%macro PROLOGUE 0
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rax, rsp                ; Save RSP
    and     rsp, 0xFFFFFFFFFFFFFFE0 ; Force 32-byte alignment (ILLEGAL on Windows without SEH)
    sub     rsp, 288
    mov     [rsp+256], rax          ; Save original RSP
    
    vmovdqa [rsp],      ymm6        ; Aligned store (Crashes if alignment fails)
    vmovdqa [rsp+32],   ymm7
    ...
```

**Fixed (Windows Compliant):**
```nasm
%macro PROLOGUE 0
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; WINDOWS FIX: Fixed stack allocation (no dynamic alignment)
    ; 56 bytes pushed + ret addr. Need 8 mod 16.
    ; 264 = 256 (regs) + 8 (padding).
    sub     rsp, 264

    vmovdqu [rsp],      ymm6        ; Unaligned store (Safe)
    vmovdqu [rsp+32],   ymm7
    vmovdqu [rsp+64],   ymm8
    vmovdqu [rsp+96],   ymm9
    vmovdqu [rsp+128],  ymm10
    vmovdqu [rsp+160],  ymm11
    vmovdqu [rsp+192],  ymm12
    vmovdqu [rsp+224],  ymm13
%endmacro
```

### EPILOGUE Macro

**Original:**
```nasm
%macro EPILOGUE 0
    vmovdqa ymm6,   [rsp]
    ...
    mov     rsp, [rsp+256]          ; Restore dynamic RSP
    pop     r15
    ...
```

**Fixed:**
```nasm
%macro EPILOGUE 0
    vmovdqu ymm6,   [rsp]           ; Unaligned load (Safe)
    vmovdqu ymm7,   [rsp+32]
    vmovdqu ymm8,   [rsp+64]
    vmovdqu ymm9,   [rsp+96]
    vmovdqu ymm10,  [rsp+128]
    vmovdqu ymm11,  [rsp+160]
    vmovdqu ymm12,  [rsp+192]
    vmovdqu ymm13,  [rsp+224]

    add     rsp, 264                ; Fixed restore
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret
%endmacro
```

## 4. Verification
This patch has been verified locally. Floating point reductions (e.g., `fp_reduce_add_f64`) that previously crashed the Erlang VM now execute correctly and return accurate results.
