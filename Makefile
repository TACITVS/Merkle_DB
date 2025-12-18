# Makefile for Windows (using MinGW/GCC + NASM)

# Paths
NIF_SRC = native/merkle_nif.c
ASM_DIR = native/fp_lib/src/asm
PRIV_DIR = priv
DLL_OUT = $(PRIV_DIR)/merkle_nif.dll

# Compiler Flags
CFLAGS = -O3 -std=c11 -Wall -shared
# Includes for Erlang NIF headers and the library headers
INCLUDES = -I"$(ERL_EI_INCLUDE_DIR)" -Inative -Inative/fp_lib/include

# NASM Flags (Win64)
NASM_FLAGS = -f win64 -I$(ASM_DIR)/

# List of ASM objects to build
ASM_OBJS = \
    native/3d_math_kernels.obj \
    native/fp_core_compaction.obj \
    native/fp_core_descriptive_stats.obj \
    native/fp_core_essentials.obj \
    native/fp_core_fused_folds_f32.obj \
    native/fp_core_fused_folds_i16.obj \
    native/fp_core_fused_folds_i32.obj \
    native/fp_core_fused_folds_i8.obj \
    native/fp_core_fused_folds_u16.obj \
    native/fp_core_fused_folds_u32.obj \
    native/fp_core_fused_folds_u64.obj \
    native/fp_core_fused_folds_u8.obj \
    native/fp_core_fused_folds.obj \
    native/fp_core_fused_maps_f32.obj \
    native/fp_core_fused_maps_i16.obj \
    native/fp_core_fused_maps_i32.obj \
    native/fp_core_fused_maps_i8.obj \
    native/fp_core_fused_maps_u16.obj \
    native/fp_core_fused_maps_u32.obj \
    native/fp_core_fused_maps_u64.obj \
    native/fp_core_fused_maps_u8.obj \
    native/fp_core_fused_maps.obj \
    native/fp_core_matrix.obj \
    native/fp_core_percentiles.obj \
    native/fp_core_predicates.obj \
    native/fp_core_reductions_f32.obj \
    native/fp_core_reductions_i16.obj \
    native/fp_core_reductions_i32.obj \
    native/fp_core_reductions_i8.obj \
    native/fp_core_reductions_u16.obj \
    native/fp_core_reductions_u32.obj \
    native/fp_core_reductions_u64.obj \
    native/fp_core_reductions_u8.obj \
    native/fp_core_reductions.obj \
    native/fp_core_scans.obj \
    native/fp_core_simple_maps.obj \
    native/fp_core_tier2.obj \
    native/fp_core_tier3.obj

# List of C objects to build
C_LIB_OBJS = \
    native/fp_compose.obj \
    native/fp_correlation_wrappers.obj \
    native/fp_general_hof.obj \
    native/fp_generic.obj \
    native/fp_monads.obj \
    native/fp_moving_averages_wrappers.obj \
    native/fp_percentile_wrappers.obj \
    native/fp_regression_wrappers.obj \
    native/fp_rolling_window.obj \
    native/fp_stats_template.obj \
    native/3d_math_wrapper.obj \
    native/fp_decision_tree.obj \
    native/fp_fft.obj \
    native/fp_gpu_math.obj \
    native/fp_kmeans.obj \
    native/fp_lighting.obj \
    native/fp_linear_regression.obj \
    native/fp_matrix_ops.obj \
    native/fp_monte_carlo.obj \
    native/fp_naive_bayes.obj \
    native/fp_neural_network.obj \
    native/fp_pca.obj \
    native/fp_quaternion_ops.obj \
    native/fp_radix_sort.obj \
    native/fp_ray_tracer.obj \
    native/fp_time_series.obj \
    native/fp_vector_ops.obj

all: $(PRIV_DIR) $(DLL_OUT)

$(PRIV_DIR):
	if not exist $(PRIV_DIR) mkdir $(PRIV_DIR)

# Generic rule for ASM files
native/%.obj: $(ASM_DIR)/%.asm
	nasm $(NASM_FLAGS) $< -o $@

# Generic rules for C files
native/%.obj: native/fp_lib/src/wrappers/%.c
	gcc $(CFLAGS) $(INCLUDES) -c $< -o $@

native/%.obj: native/fp_lib/src/algorithms/%.c
	gcc $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link C NIF + ASM Objects + C Lib Objects into DLL
$(DLL_OUT): $(ASM_OBJS) $(C_LIB_OBJS) $(NIF_SRC)
	gcc $(CFLAGS) $(INCLUDES) -o $(DLL_OUT) $(NIF_SRC) $(ASM_OBJS) $(C_LIB_OBJS)

clean:
	del native\*.obj
	del priv\*.dll
