// GENERATED V7 - ZERO-COPY
#include <stdbool.h>
#include <string.h>
#ifdef __GNUC__
  #define _SAVED_GNUC_ __GNUC__ 
  #undef __GNUC__
#endif
#include <erl_nif.h>
#ifdef _SAVED_GNUC_
  #define __GNUC__ _SAVED_GNUC__
  #undef _SAVED_GNUC_
#endif
#include "fp_lib/include/fp_core.h"
#include "fp_lib/include/fp_stats.h"
#include "fp_lib/include/fp_linear_regression.h"
#include "fp_lib/include/fp_monads.h"
#include "fp_lib/include/fp_compose.h"
#include "fp_lib/include/fp_3d_math_wrapper.h"
#include "fp_lib/include/fp_gpu_math.h"
#include "fp_lib/include/fp_math.h"
#include "fp_lib/include/fp_pca.h"
#include "fp_lib/include/fp_kmeans.h"
#include "fp_lib/include/fp_naive_bayes.h"
#include "fp_lib/include/fp_neural_network.h"
ErlNifResourceType* RES_TYPE_GaussianNBModel;
ErlNifResourceType* RES_TYPE_KMeansResult;
ErlNifResourceType* RES_TYPE_MultinomialNBModel;
ErlNifResourceType* RES_TYPE_NeuralNetwork;
ErlNifResourceType* RES_TYPE_PCAModel;
ErlNifResourceType* RES_TYPE_PCAResult;
ErlNifResourceType* RES_TYPE_TrainingResult;
static void fp_pca_free_result_internal(PCAResult* res) { fp_pca_free_model(&res->model); }
void dtor_GaussianNBModel(ErlNifEnv* env, void* obj) { GaussianNBModel* res = (GaussianNBModel*)obj; fp_nb_free_gaussian_model(res); }

void dtor_KMeansResult(ErlNifEnv* env, void* obj) { KMeansResult* res = (KMeansResult*)obj; fp_kmeans_free(res); }

void dtor_MultinomialNBModel(ErlNifEnv* env, void* obj) { MultinomialNBModel* res = (MultinomialNBModel*)obj; fp_nb_free_multinomial_model(res); }

void dtor_NeuralNetwork(ErlNifEnv* env, void* obj) { NeuralNetwork* res = (NeuralNetwork*)obj; fp_neural_network_free(res); }

void dtor_PCAModel(ErlNifEnv* env, void* obj) { PCAModel* res = (PCAModel*)obj; fp_pca_free_model(res); }

void dtor_PCAResult(ErlNifEnv* env, void* obj) { PCAResult* res = (PCAResult*)obj; fp_pca_free_result_internal(res); }

void dtor_TrainingResult(ErlNifEnv* env, void* obj) { TrainingResult* res = (TrainingResult*)obj; fp_training_result_free(res); }

static ERL_NIF_TERM nif_fp_concat_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input_a; if (!enif_inspect_binary(env, argv[0], &bin_input_a)) return enif_make_badarg(env); int64_t* ptr_input_a = (int64_t*)bin_input_a.data;
    ErlNifBinary bin_input_b; if (!enif_inspect_binary(env, argv[1], &bin_input_b)) return enif_make_badarg(env); int64_t* ptr_input_b = (int64_t*)bin_input_b.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[2], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_len_a; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_len_a)) return enif_make_badarg(env);
    size_t val_len_b; if (!enif_get_uint64(env, argv[4], (ErlNifUInt64*)&val_len_b)) return enif_make_badarg(env);
    size_t res = fp_concat_i64(ptr_input_a, ptr_input_b, ptr_output, val_len_a, val_len_b);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_contains_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_target; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_target)) return enif_make_badarg(env);
    bool res = fp_contains_i64(ptr_input, val_n, val_target);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_correlation_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); double* ptr_x = (double*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); double* ptr_y = (double*)bin_y.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_correlation_f64(ptr_x, ptr_y, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_count_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_target; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_target)) return enif_make_badarg(env);
    size_t res = fp_count_i64(ptr_input, val_n, val_target);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_covariance_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); double* ptr_x = (double*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); double* ptr_y = (double*)bin_y.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_covariance_f64(ptr_x, ptr_y, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_detect_outliers_iqr_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_k_factor; if (!enif_get_double(env, argv[2], &val_k_factor)) return enif_make_badarg(env);
    ErlNifUInt64 size_is_outlier; if (!enif_get_uint64(env, argv[3], &size_is_outlier)) return enif_make_badarg(env); ErlNifBinary out_bin_is_outlier; enif_alloc_binary((size_t)size_is_outlier, &out_bin_is_outlier); uint8_t* ptr_is_outlier = (uint8_t*)out_bin_is_outlier.data;
    size_t res = fp_detect_outliers_iqr_f64(ptr_data, val_n, val_k_factor, ptr_is_outlier);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_is_outlier));
}

static ERL_NIF_TERM nif_fp_detect_outliers_zscore_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_threshold; if (!enif_get_double(env, argv[2], &val_threshold)) return enif_make_badarg(env);
    ErlNifUInt64 size_is_outlier; if (!enif_get_uint64(env, argv[3], &size_is_outlier)) return enif_make_badarg(env); ErlNifBinary out_bin_is_outlier; enif_alloc_binary((size_t)size_is_outlier, &out_bin_is_outlier); uint8_t* ptr_is_outlier = (uint8_t*)out_bin_is_outlier.data;
    size_t res = fp_detect_outliers_zscore_f64(ptr_data, val_n, val_threshold, ptr_is_outlier);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_is_outlier));
}

static ERL_NIF_TERM nif_fp_drop_n_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_array_len; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_array_len)) return enif_make_badarg(env);
    size_t val_drop_count; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_drop_count)) return enif_make_badarg(env);
    size_t res = fp_drop_n_i64(ptr_input, ptr_output, val_array_len, val_drop_count);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_drop_while_gt_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_threshold; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_threshold)) return enif_make_badarg(env);
    size_t res = fp_drop_while_gt_i64(ptr_input, ptr_output, val_n, val_threshold);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_ema_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_ema_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_filter_gt_i64_simple(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_threshold; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_threshold)) return enif_make_badarg(env);
    size_t res = fp_filter_gt_i64_simple(ptr_input, ptr_output, val_n, val_threshold);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_find_index_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_target; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_target)) return enif_make_badarg(env);
    int64_t res = fp_find_index_i64(ptr_input, val_n, val_target);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); float* ptr_a = (float*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); float* ptr_b = (float*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_fold_dotp_f32(ptr_a, ptr_b, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); double* ptr_a = (double*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); double* ptr_b = (double*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_fold_dotp_f64(ptr_a, ptr_b, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int16_t* ptr_a = (int16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int16_t* ptr_b = (int16_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_fold_dotp_i16(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_dotp_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int32_t* ptr_a = (int32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int32_t* ptr_b = (int32_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_fold_dotp_i32(ptr_a, ptr_b, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int64_t* ptr_a = (int64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int64_t* ptr_b = (int64_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_fold_dotp_i64(ptr_a, ptr_b, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int8_t* ptr_a = (int8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int8_t* ptr_b = (int8_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_fold_dotp_i8(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_dotp_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint16_t* ptr_a = (uint16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint16_t* ptr_b = (uint16_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_fold_dotp_u16(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_dotp_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint32_t* ptr_a = (uint32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint32_t* ptr_b = (uint32_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_fold_dotp_u32(ptr_a, ptr_b, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint64_t* ptr_a = (uint64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint64_t* ptr_b = (uint64_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_fold_dotp_u64(ptr_a, ptr_b, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_dotp_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint8_t* ptr_a = (uint8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint8_t* ptr_b = (uint8_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_fold_dotp_u8(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sad_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); float* ptr_a = (float*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); float* ptr_b = (float*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_fold_sad_f32(ptr_a, ptr_b, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_fold_sad_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int16_t* ptr_a = (int16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int16_t* ptr_b = (int16_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_fold_sad_i16(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sad_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int32_t* ptr_a = (int32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int32_t* ptr_b = (int32_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_fold_sad_i32(ptr_a, ptr_b, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sad_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int64_t* ptr_a = (int64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int64_t* ptr_b = (int64_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_fold_sad_i64(ptr_a, ptr_b, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sad_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int8_t* ptr_a = (int8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int8_t* ptr_b = (int8_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_fold_sad_i8(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sad_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint16_t* ptr_a = (uint16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint16_t* ptr_b = (uint16_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_fold_sad_u16(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sad_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint32_t* ptr_a = (uint32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint32_t* ptr_b = (uint32_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_fold_sad_u32(ptr_a, ptr_b, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sad_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint64_t* ptr_a = (uint64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint64_t* ptr_b = (uint64_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_fold_sad_u64(ptr_a, ptr_b, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sad_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint8_t* ptr_a = (uint8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint8_t* ptr_b = (uint8_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_fold_sad_u8(ptr_a, ptr_b, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_fold_sumsq_f32(ptr_in, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_fold_sumsq_i16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_fold_sumsq_i32(ptr_in, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_fold_sumsq_i64(ptr_in, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_fold_sumsq_i8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_fold_sumsq_u16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_fold_sumsq_u32(ptr_in, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_fold_sumsq_u64(ptr_in, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_fold_sumsq_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_fold_sumsq_u8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_gaussian_nb_predict_batch(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    GaussianNBModel* res_model; if (!enif_get_resource(env, argv[0], RES_TYPE_GaussianNBModel, (void**)&res_model)) return enif_make_badarg(env);
    ErlNifBinary bin_X; if (!enif_inspect_binary(env, argv[1], &bin_X)) return enif_make_badarg(env); double* ptr_X = (double*)bin_X.data;
    int val_n; if (!enif_get_int(env, argv[2], (int*)&val_n)) return enif_make_badarg(env);
    ErlNifUInt64 size_predictions; if (!enif_get_uint64(env, argv[3], &size_predictions)) return enif_make_badarg(env); ErlNifBinary out_bin_predictions; enif_alloc_binary((size_t)size_predictions, &out_bin_predictions); int* ptr_predictions = (int*)out_bin_predictions.data;
    fp_gaussian_nb_predict_batch(res_model, ptr_X, val_n, ptr_predictions);
    return enif_make_binary(env, &out_bin_predictions);
}

static ERL_NIF_TERM nif_fp_gaussian_nb_train(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_X; if (!enif_inspect_binary(env, argv[0], &bin_X)) return enif_make_badarg(env); double* ptr_X = (double*)bin_X.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int* ptr_y = (int*)bin_y.data;
    int val_n; if (!enif_get_int(env, argv[2], (int*)&val_n)) return enif_make_badarg(env);
    int val_d; if (!enif_get_int(env, argv[3], (int*)&val_d)) return enif_make_badarg(env);
    int val_n_classes; if (!enif_get_int(env, argv[4], (int*)&val_n_classes)) return enif_make_badarg(env);
    GaussianNBModel res = fp_gaussian_nb_train(ptr_X, ptr_y, val_n, val_d, val_n_classes);
    GaussianNBModel* res_ptr = enif_alloc_resource(RES_TYPE_GaussianNBModel, sizeof(GaussianNBModel));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_group_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_groups_out; if (!enif_get_uint64(env, argv[1], &size_groups_out)) return enif_make_badarg(env); ErlNifBinary out_bin_groups_out; enif_alloc_binary((size_t)size_groups_out, &out_bin_groups_out); int64_t* ptr_groups_out = (int64_t*)out_bin_groups_out.data;
    ErlNifUInt64 size_counts_out; if (!enif_get_uint64(env, argv[2], &size_counts_out)) return enif_make_badarg(env); ErlNifBinary out_bin_counts_out; enif_alloc_binary((size_t)size_counts_out, &out_bin_counts_out); int64_t* ptr_counts_out = (int64_t*)out_bin_counts_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t res = fp_group_i64(ptr_input, ptr_groups_out, ptr_counts_out, val_n);
    return enif_make_tuple3(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_groups_out), enif_make_binary(env, &out_bin_counts_out));
}

static ERL_NIF_TERM nif_fp_intersect_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_array_a; if (!enif_inspect_binary(env, argv[0], &bin_array_a)) return enif_make_badarg(env); int64_t* ptr_array_a = (int64_t*)bin_array_a.data;
    ErlNifBinary bin_array_b; if (!enif_inspect_binary(env, argv[1], &bin_array_b)) return enif_make_badarg(env); int64_t* ptr_array_b = (int64_t*)bin_array_b.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[2], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_len_a; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_len_a)) return enif_make_badarg(env);
    size_t val_len_b; if (!enif_get_uint64(env, argv[4], (ErlNifUInt64*)&val_len_b)) return enif_make_badarg(env);
    size_t res = fp_intersect_i64(ptr_array_a, ptr_array_b, ptr_output, val_len_a, val_len_b);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_iterate_add_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[0], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_start; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_start)) return enif_make_badarg(env);
    int64_t val_step; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_step)) return enif_make_badarg(env);
    fp_iterate_add_i64(ptr_output, val_n, val_start, val_step);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_iterate_mul_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[0], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_start; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_start)) return enif_make_badarg(env);
    int64_t val_factor; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_factor)) return enif_make_badarg(env);
    fp_iterate_mul_i64(ptr_output, val_n, val_start, val_factor);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_kmeans_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    int val_n; if (!enif_get_int(env, argv[1], (int*)&val_n)) return enif_make_badarg(env);
    int val_d; if (!enif_get_int(env, argv[2], (int*)&val_d)) return enif_make_badarg(env);
    int val_k; if (!enif_get_int(env, argv[3], (int*)&val_k)) return enif_make_badarg(env);
    int val_max_iter; if (!enif_get_int(env, argv[4], (int*)&val_max_iter)) return enif_make_badarg(env);
    double val_tol; if (!enif_get_double(env, argv[5], &val_tol)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[6], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    KMeansResult res = fp_kmeans_f64(ptr_data, val_n, val_d, val_k, val_max_iter, val_tol, val_seed);
    KMeansResult* res_ptr = enif_alloc_resource(RES_TYPE_KMeansResult, sizeof(KMeansResult));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_linear_regression_r2_score(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_y_true; if (!enif_inspect_binary(env, argv[0], &bin_y_true)) return enif_make_badarg(env); double* ptr_y_true = (double*)bin_y_true.data;
    ErlNifBinary bin_y_pred; if (!enif_inspect_binary(env, argv[1], &bin_y_pred)) return enif_make_badarg(env); double* ptr_y_pred = (double*)bin_y_pred.data;
    int val_n; if (!enif_get_int(env, argv[2], (int*)&val_n)) return enif_make_badarg(env);
    double res = fp_linear_regression_r2_score(ptr_y_true, ptr_y_pred, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_map_abs_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_map_abs_f64(ptr_in, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_abs_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_map_abs_i64(ptr_in, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); float* ptr_x = (float*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); float* ptr_y = (float*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); float* ptr_out = (float*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float val_c; double tmp_4; if (!enif_get_double(env, argv[4], &tmp_4)) return enif_make_badarg(env); val_c = (float)tmp_4;
    fp_map_axpy_f32(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); double* ptr_x = (double*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); double* ptr_y = (double*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_c; if (!enif_get_double(env, argv[4], &val_c)) return enif_make_badarg(env);
    fp_map_axpy_f64(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); int16_t* ptr_x = (int16_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int16_t* ptr_y = (int16_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int16_t* ptr_out = (int16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t val_c; int tmp_4; if (!enif_get_int(env, argv[4], &tmp_4)) return enif_make_badarg(env); val_c = (int16_t)tmp_4;
    fp_map_axpy_i16(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); int32_t* ptr_x = (int32_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int32_t* ptr_y = (int32_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int32_t* ptr_out = (int32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t val_c; if (!enif_get_int(env, argv[4], (int*)&val_c)) return enif_make_badarg(env);
    fp_map_axpy_i32(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); int64_t* ptr_x = (int64_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int64_t* ptr_y = (int64_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_c; if (!enif_get_int64(env, argv[4], (ErlNifSInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_axpy_i64(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); int8_t* ptr_x = (int8_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int8_t* ptr_y = (int8_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int8_t* ptr_out = (int8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t val_c; int tmp_4; if (!enif_get_int(env, argv[4], &tmp_4)) return enif_make_badarg(env); val_c = (int8_t)tmp_4;
    fp_map_axpy_i8(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); uint16_t* ptr_x = (uint16_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); uint16_t* ptr_y = (uint16_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint16_t* ptr_out = (uint16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t val_c; unsigned int tmp_4; if (!enif_get_uint(env, argv[4], &tmp_4)) return enif_make_badarg(env); val_c = (uint16_t)tmp_4;
    fp_map_axpy_u16(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); uint32_t* ptr_x = (uint32_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); uint32_t* ptr_y = (uint32_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint32_t* ptr_out = (uint32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t val_c; if (!enif_get_uint(env, argv[4], (unsigned int*)&val_c)) return enif_make_badarg(env);
    fp_map_axpy_u32(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); uint64_t* ptr_x = (uint64_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); uint64_t* ptr_y = (uint64_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint64_t* ptr_out = (uint64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t val_c; if (!enif_get_uint64(env, argv[4], (ErlNifUInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_axpy_u64(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_axpy_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); uint8_t* ptr_x = (uint8_t*)bin_x.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); uint8_t* ptr_y = (uint8_t*)bin_y.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint8_t* ptr_out = (uint8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t val_c; unsigned int tmp_4; if (!enif_get_uint(env, argv[4], &tmp_4)) return enif_make_badarg(env); val_c = (uint8_t)tmp_4;
    fp_map_axpy_u8(ptr_x, ptr_y, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_clamp_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_min_val; if (!enif_get_double(env, argv[3], &val_min_val)) return enif_make_badarg(env);
    double val_max_val; if (!enif_get_double(env, argv[4], &val_max_val)) return enif_make_badarg(env);
    fp_map_clamp_f64(ptr_in, ptr_out, val_n, val_min_val, val_max_val);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_clamp_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_min_val; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_min_val)) return enif_make_badarg(env);
    int64_t val_max_val; if (!enif_get_int64(env, argv[4], (ErlNifSInt64*)&val_max_val)) return enif_make_badarg(env);
    fp_map_clamp_i64(ptr_in, ptr_out, val_n, val_min_val, val_max_val);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); float* ptr_out = (float*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float val_c; double tmp_3; if (!enif_get_double(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (float)tmp_3;
    fp_map_offset_f32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_c; if (!enif_get_double(env, argv[3], &val_c)) return enif_make_badarg(env);
    fp_map_offset_f64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int16_t* ptr_out = (int16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t val_c; int tmp_3; if (!enif_get_int(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (int16_t)tmp_3;
    fp_map_offset_i16(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int32_t* ptr_out = (int32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t val_c; if (!enif_get_int(env, argv[3], (int*)&val_c)) return enif_make_badarg(env);
    fp_map_offset_i32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_c; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_offset_i64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int8_t* ptr_out = (int8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t val_c; int tmp_3; if (!enif_get_int(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (int8_t)tmp_3;
    fp_map_offset_i8(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint16_t* ptr_out = (uint16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t val_c; unsigned int tmp_3; if (!enif_get_uint(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (uint16_t)tmp_3;
    fp_map_offset_u16(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint32_t* ptr_out = (uint32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t val_c; if (!enif_get_uint(env, argv[3], (unsigned int*)&val_c)) return enif_make_badarg(env);
    fp_map_offset_u32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint64_t* ptr_out = (uint64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t val_c; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_offset_u64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_offset_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint8_t* ptr_out = (uint8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t val_c; unsigned int tmp_3; if (!enif_get_uint(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (uint8_t)tmp_3;
    fp_map_offset_u8(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); float* ptr_out = (float*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float val_c; double tmp_3; if (!enif_get_double(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (float)tmp_3;
    fp_map_scale_f32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_c; if (!enif_get_double(env, argv[3], &val_c)) return enif_make_badarg(env);
    fp_map_scale_f64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int16_t* ptr_out = (int16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t val_c; int tmp_3; if (!enif_get_int(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (int16_t)tmp_3;
    fp_map_scale_i16(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int32_t* ptr_out = (int32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t val_c; if (!enif_get_int(env, argv[3], (int*)&val_c)) return enif_make_badarg(env);
    fp_map_scale_i32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_c; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_scale_i64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int8_t* ptr_out = (int8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t val_c; int tmp_3; if (!enif_get_int(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (int8_t)tmp_3;
    fp_map_scale_i8(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint16_t* ptr_out = (uint16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t val_c; unsigned int tmp_3; if (!enif_get_uint(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (uint16_t)tmp_3;
    fp_map_scale_u16(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint32_t* ptr_out = (uint32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t val_c; if (!enif_get_uint(env, argv[3], (unsigned int*)&val_c)) return enif_make_badarg(env);
    fp_map_scale_u32(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint64_t* ptr_out = (uint64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t val_c; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_c)) return enif_make_badarg(env);
    fp_map_scale_u64(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_scale_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint8_t* ptr_out = (uint8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t val_c; unsigned int tmp_3; if (!enif_get_uint(env, argv[3], &tmp_3)) return enif_make_badarg(env); val_c = (uint8_t)tmp_3;
    fp_map_scale_u8(ptr_in, ptr_out, val_n, val_c);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_map_sqrt_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_map_sqrt_f64(ptr_in, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_moments_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    ErlNifUInt64 size_moments; if (!enif_get_uint64(env, argv[2], &size_moments)) return enif_make_badarg(env); ErlNifBinary out_bin_moments; enif_alloc_binary((size_t)size_moments, &out_bin_moments); double* ptr_moments = (double*)out_bin_moments.data;
    fp_moments_f64(ptr_data, val_n, ptr_moments);
    return enif_make_binary(env, &out_bin_moments);
}

static ERL_NIF_TERM nif_fp_multinomial_nb_predict_batch(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MultinomialNBModel* res_model; if (!enif_get_resource(env, argv[0], RES_TYPE_MultinomialNBModel, (void**)&res_model)) return enif_make_badarg(env);
    ErlNifBinary bin_X; if (!enif_inspect_binary(env, argv[1], &bin_X)) return enif_make_badarg(env); double* ptr_X = (double*)bin_X.data;
    int val_n; if (!enif_get_int(env, argv[2], (int*)&val_n)) return enif_make_badarg(env);
    ErlNifUInt64 size_predictions; if (!enif_get_uint64(env, argv[3], &size_predictions)) return enif_make_badarg(env); ErlNifBinary out_bin_predictions; enif_alloc_binary((size_t)size_predictions, &out_bin_predictions); int* ptr_predictions = (int*)out_bin_predictions.data;
    fp_multinomial_nb_predict_batch(res_model, ptr_X, val_n, ptr_predictions);
    return enif_make_binary(env, &out_bin_predictions);
}

static ERL_NIF_TERM nif_fp_multinomial_nb_train(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_X; if (!enif_inspect_binary(env, argv[0], &bin_X)) return enif_make_badarg(env); double* ptr_X = (double*)bin_X.data;
    ErlNifBinary bin_y; if (!enif_inspect_binary(env, argv[1], &bin_y)) return enif_make_badarg(env); int* ptr_y = (int*)bin_y.data;
    int val_n; if (!enif_get_int(env, argv[2], (int*)&val_n)) return enif_make_badarg(env);
    int val_d; if (!enif_get_int(env, argv[3], (int*)&val_d)) return enif_make_badarg(env);
    int val_n_classes; if (!enif_get_int(env, argv[4], (int*)&val_n_classes)) return enif_make_badarg(env);
    double val_alpha; if (!enif_get_double(env, argv[5], &val_alpha)) return enif_make_badarg(env);
    MultinomialNBModel res = fp_multinomial_nb_train(ptr_X, ptr_y, val_n, val_d, val_n_classes, val_alpha);
    MultinomialNBModel* res_ptr = enif_alloc_resource(RES_TYPE_MultinomialNBModel, sizeof(MultinomialNBModel));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_neural_network_create(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    int val_n_inputs; if (!enif_get_int(env, argv[0], (int*)&val_n_inputs)) return enif_make_badarg(env);
    int val_n_hidden; if (!enif_get_int(env, argv[1], (int*)&val_n_hidden)) return enif_make_badarg(env);
    int val_n_outputs; if (!enif_get_int(env, argv[2], (int*)&val_n_outputs)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    NeuralNetwork res = fp_neural_network_create(val_n_inputs, val_n_hidden, val_n_outputs, val_seed);
    NeuralNetwork* res_ptr = enif_alloc_resource(RES_TYPE_NeuralNetwork, sizeof(NeuralNetwork));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_neural_network_print_summary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    NeuralNetwork* res_net; if (!enif_get_resource(env, argv[0], RES_TYPE_NeuralNetwork, (void**)&res_net)) return enif_make_badarg(env);
    fp_neural_network_print_summary(res_net);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_fp_neural_network_train(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    int val_n_inputs; if (!enif_get_int(env, argv[0], (int*)&val_n_inputs)) return enif_make_badarg(env);
    int val_n_hidden; if (!enif_get_int(env, argv[1], (int*)&val_n_hidden)) return enif_make_badarg(env);
    int val_n_outputs; if (!enif_get_int(env, argv[2], (int*)&val_n_outputs)) return enif_make_badarg(env);
    ErlNifBinary bin_X_train; if (!enif_inspect_binary(env, argv[3], &bin_X_train)) return enif_make_badarg(env); double* ptr_X_train = (double*)bin_X_train.data;
    ErlNifBinary bin_y_train; if (!enif_inspect_binary(env, argv[4], &bin_y_train)) return enif_make_badarg(env); double* ptr_y_train = (double*)bin_y_train.data;
    int val_n_samples; if (!enif_get_int(env, argv[5], (int*)&val_n_samples)) return enif_make_badarg(env);
    int val_n_epochs; if (!enif_get_int(env, argv[6], (int*)&val_n_epochs)) return enif_make_badarg(env);
    double val_learning_rate; if (!enif_get_double(env, argv[7], &val_learning_rate)) return enif_make_badarg(env);
    int val_verbose; if (!enif_get_int(env, argv[8], (int*)&val_verbose)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[9], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    TrainingResult res = fp_neural_network_train(val_n_inputs, val_n_hidden, val_n_outputs, ptr_X_train, ptr_y_train, val_n_samples, val_n_epochs, val_learning_rate, val_verbose, val_seed);
    TrainingResult* res_ptr = enif_alloc_resource(RES_TYPE_TrainingResult, sizeof(TrainingResult));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_partition_gt_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output_pass; if (!enif_get_uint64(env, argv[1], &size_output_pass)) return enif_make_badarg(env); ErlNifBinary out_bin_output_pass; enif_alloc_binary((size_t)size_output_pass, &out_bin_output_pass); int64_t* ptr_output_pass = (int64_t*)out_bin_output_pass.data;
    ErlNifUInt64 size_output_fail; if (!enif_get_uint64(env, argv[2], &size_output_fail)) return enif_make_badarg(env); ErlNifBinary out_bin_output_fail; enif_alloc_binary((size_t)size_output_fail, &out_bin_output_fail); int64_t* ptr_output_fail = (int64_t*)out_bin_output_fail.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_threshold; if (!enif_get_int64(env, argv[4], (ErlNifSInt64*)&val_threshold)) return enif_make_badarg(env);
    ErlNifUInt64 size_out_pass_count; if (!enif_get_uint64(env, argv[5], &size_out_pass_count)) return enif_make_badarg(env); ErlNifBinary out_bin_out_pass_count; enif_alloc_binary((size_t)size_out_pass_count, &out_bin_out_pass_count); size_t* ptr_out_pass_count = (size_t*)out_bin_out_pass_count.data;
    ErlNifUInt64 size_out_fail_count; if (!enif_get_uint64(env, argv[6], &size_out_fail_count)) return enif_make_badarg(env); ErlNifBinary out_bin_out_fail_count; enif_alloc_binary((size_t)size_out_fail_count, &out_bin_out_fail_count); size_t* ptr_out_fail_count = (size_t*)out_bin_out_fail_count.data;
    fp_partition_gt_i64(ptr_input, ptr_output_pass, ptr_output_fail, val_n, val_threshold, ptr_out_pass_count, ptr_out_fail_count);
    return enif_make_tuple4(env, enif_make_binary(env, &out_bin_output_pass), enif_make_binary(env, &out_bin_output_fail), enif_make_binary(env, &out_bin_out_pass_count), enif_make_binary(env, &out_bin_out_fail_count));
}

static ERL_NIF_TERM nif_fp_pca_fit(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_X; if (!enif_inspect_binary(env, argv[0], &bin_X)) return enif_make_badarg(env); double* ptr_X = (double*)bin_X.data;
    int val_n; if (!enif_get_int(env, argv[1], (int*)&val_n)) return enif_make_badarg(env);
    int val_d; if (!enif_get_int(env, argv[2], (int*)&val_d)) return enif_make_badarg(env);
    int val_n_components; if (!enif_get_int(env, argv[3], (int*)&val_n_components)) return enif_make_badarg(env);
    int val_max_iterations; if (!enif_get_int(env, argv[4], (int*)&val_max_iterations)) return enif_make_badarg(env);
    double val_tolerance; if (!enif_get_double(env, argv[5], &val_tolerance)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[6], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    PCAResult res = fp_pca_fit(ptr_X, val_n, val_d, val_n_components, val_max_iterations, val_tolerance, val_seed);
    PCAResult* res_ptr = enif_alloc_resource(RES_TYPE_PCAResult, sizeof(PCAResult));
    *res_ptr = res;
    ERL_NIF_TERM ret_res = enif_make_resource(env, res_ptr);
    enif_release_resource(res_ptr);

    return ret_res;
}

static ERL_NIF_TERM nif_fp_pca_generate_ellipse_data(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_X; if (!enif_get_uint64(env, argv[0], &size_X)) return enif_make_badarg(env); ErlNifBinary out_bin_X; enif_alloc_binary((size_t)size_X, &out_bin_X); double* ptr_X = (double*)out_bin_X.data;
    int val_n; if (!enif_get_int(env, argv[1], (int*)&val_n)) return enif_make_badarg(env);
    double val_major_axis; if (!enif_get_double(env, argv[2], &val_major_axis)) return enif_make_badarg(env);
    double val_minor_axis; if (!enif_get_double(env, argv[3], &val_minor_axis)) return enif_make_badarg(env);
    double val_angle; if (!enif_get_double(env, argv[4], &val_angle)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[5], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    fp_pca_generate_ellipse_data(ptr_X, val_n, val_major_axis, val_minor_axis, val_angle, val_seed);
    return enif_make_binary(env, &out_bin_X);
}

static ERL_NIF_TERM nif_fp_pca_generate_low_rank_data(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_X; if (!enif_get_uint64(env, argv[0], &size_X)) return enif_make_badarg(env); ErlNifBinary out_bin_X; enif_alloc_binary((size_t)size_X, &out_bin_X); double* ptr_X = (double*)out_bin_X.data;
    int val_n; if (!enif_get_int(env, argv[1], (int*)&val_n)) return enif_make_badarg(env);
    int val_d; if (!enif_get_int(env, argv[2], (int*)&val_d)) return enif_make_badarg(env);
    int val_intrinsic_dim; if (!enif_get_int(env, argv[3], (int*)&val_intrinsic_dim)) return enif_make_badarg(env);
    double val_noise_stddev; if (!enif_get_double(env, argv[4], &val_noise_stddev)) return enif_make_badarg(env);
    uint64_t val_seed; if (!enif_get_uint64(env, argv[5], (ErlNifUInt64*)&val_seed)) return enif_make_badarg(env);
    fp_pca_generate_low_rank_data(ptr_X, val_n, val_d, val_intrinsic_dim, val_noise_stddev, val_seed);
    return enif_make_binary(env, &out_bin_X);
}

static ERL_NIF_TERM nif_fp_percentile_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_p; if (!enif_get_double(env, argv[2], &val_p)) return enif_make_badarg(env);
    double res = fp_percentile_f64(ptr_data, val_n, val_p);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_percentiles_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    ErlNifBinary bin_p_values; if (!enif_inspect_binary(env, argv[2], &bin_p_values)) return enif_make_badarg(env); double* ptr_p_values = (double*)bin_p_values.data;
    size_t val_n_percentiles; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n_percentiles)) return enif_make_badarg(env);
    ErlNifUInt64 size_results; if (!enif_get_uint64(env, argv[4], &size_results)) return enif_make_badarg(env); ErlNifBinary out_bin_results; enif_alloc_binary((size_t)size_results, &out_bin_results); double* ptr_results = (double*)out_bin_results.data;
    fp_percentiles_f64(ptr_data, val_n, ptr_p_values, val_n_percentiles, ptr_results);
    return enif_make_binary(env, &out_bin_results);
}

static ERL_NIF_TERM nif_fp_pred_all_eq_const_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_arr; if (!enif_inspect_binary(env, argv[0], &bin_arr)) return enif_make_badarg(env); int64_t* ptr_arr = (int64_t*)bin_arr.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_value; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_value)) return enif_make_badarg(env);
    bool res = fp_pred_all_eq_const_i64(ptr_arr, val_n, val_value);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_pred_all_gt_zip_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int64_t* ptr_a = (int64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int64_t* ptr_b = (int64_t*)bin_b.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    bool res = fp_pred_all_gt_zip_i64(ptr_a, ptr_b, val_n);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_pred_any_gt_const_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_arr; if (!enif_inspect_binary(env, argv[0], &bin_arr)) return enif_make_badarg(env); int64_t* ptr_arr = (int64_t*)bin_arr.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_value; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_value)) return enif_make_badarg(env);
    bool res = fp_pred_any_gt_const_i64(ptr_arr, val_n, val_value);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_range_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[0], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    int64_t val_start; if (!enif_get_int64(env, argv[1], (ErlNifSInt64*)&val_start)) return enif_make_badarg(env);
    int64_t val_end; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_end)) return enif_make_badarg(env);
    size_t res = fp_range_i64(ptr_output, val_start, val_end);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_reduce_add_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_reduce_add_f32(ptr_in, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_reduce_add_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_reduce_add_f64(ptr_in, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_f64_where(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_x; if (!enif_inspect_binary(env, argv[0], &bin_x)) return enif_make_badarg(env); double* ptr_x = (double*)bin_x.data;
    ErlNifBinary bin_mask; if (!enif_inspect_binary(env, argv[1], &bin_mask)) return enif_make_badarg(env); int* ptr_mask = (int*)bin_mask.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_reduce_add_f64_where(ptr_x, ptr_mask, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_reduce_add_i16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_add_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_reduce_add_i32(ptr_in, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_reduce_add_i64(ptr_in, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_reduce_add_i8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_add_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_reduce_add_u16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_add_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_reduce_add_u32(ptr_in, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_reduce_add_u64(ptr_in, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_add_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_reduce_add_u8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_and_bool(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    bool res = fp_reduce_and_bool(ptr_input, val_n);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_reduce_max_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_reduce_max_f32(ptr_in, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_reduce_max_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_reduce_max_f64(ptr_in, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_max_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_reduce_max_i16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_max_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_reduce_max_i32(ptr_in, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_max_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_reduce_max_i64(ptr_in, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_max_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_reduce_max_i8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_max_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_reduce_max_u16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_max_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_reduce_max_u32(ptr_in, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_max_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_reduce_max_u64(ptr_in, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_max_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_reduce_max_u8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_min_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_reduce_min_f32(ptr_in, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_reduce_min_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_reduce_min_f64(ptr_in, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_min_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_reduce_min_i16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_min_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_reduce_min_i32(ptr_in, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_min_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_reduce_min_i64(ptr_in, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_min_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_reduce_min_i8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_min_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_reduce_min_u16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_min_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_reduce_min_u32(ptr_in, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_min_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_reduce_min_u64(ptr_in, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_min_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_reduce_min_u8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_mul_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); float* ptr_in = (float*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    float res = fp_reduce_mul_f32(ptr_in, val_n);
    return enif_make_double(env, (double)res);
}

static ERL_NIF_TERM nif_fp_reduce_mul_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int16_t* ptr_in = (int16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int16_t res = fp_reduce_mul_i16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_mul_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int32_t* ptr_in = (int32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int32_t res = fp_reduce_mul_i32(ptr_in, val_n);
    return enif_make_int(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_mul_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int8_t* ptr_in = (int8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int8_t res = fp_reduce_mul_i8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_mul_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint16_t* ptr_in = (uint16_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint16_t res = fp_reduce_mul_u16(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_mul_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint32_t* ptr_in = (uint32_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint32_t res = fp_reduce_mul_u32(ptr_in, val_n);
    return enif_make_uint(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_mul_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint64_t* ptr_in = (uint64_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint64_t res = fp_reduce_mul_u64(ptr_in, val_n);
    return enif_make_uint64(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_mul_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); uint8_t* ptr_in = (uint8_t*)bin_in.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    uint8_t res = fp_reduce_mul_u8(ptr_in, val_n);
    return enif_make_int(env, 0);
}

static ERL_NIF_TERM nif_fp_reduce_or_bool(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    bool res = fp_reduce_or_bool(ptr_input, val_n);
    return res ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

static ERL_NIF_TERM nif_fp_reduce_product_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); double* ptr_input = (double*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double res = fp_reduce_product_f64(ptr_input, val_n);
    return enif_make_double(env, res);
}

static ERL_NIF_TERM nif_fp_reduce_product_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t res = fp_reduce_product_i64(ptr_input, val_n);
    return enif_make_int64(env, res);
}

static ERL_NIF_TERM nif_fp_replicate_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[0], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    double val_value; if (!enif_get_double(env, argv[2], &val_value)) return enif_make_badarg(env);
    fp_replicate_f64(ptr_output, val_n, val_value);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_replicate_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[0], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_value; if (!enif_get_int64(env, argv[2], (ErlNifSInt64*)&val_value)) return enif_make_badarg(env);
    fp_replicate_i64(ptr_output, val_n, val_value);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_reverse_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_reverse_i64(ptr_input, ptr_output, val_n);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_max_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_max_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_max_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); int64_t* ptr_data = (int64_t*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    fp_rolling_max_i64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_mean_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_mean_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_mean_f64_optimized(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_mean_f64_optimized(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_min_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_min_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_min_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); int64_t* ptr_data = (int64_t*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    fp_rolling_min_i64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_range_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_range_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_std_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_std_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_sum_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_sum_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_sum_f64_optimized(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_sum_f64_optimized(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_sum_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); int64_t* ptr_data = (int64_t*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    fp_rolling_sum_i64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_rolling_variance_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_rolling_variance_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_run_length_encode_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t res = fp_run_length_encode_i64(ptr_input, ptr_output, val_n);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_scan_add_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); double* ptr_in = (double*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_scan_add_f64(ptr_in, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_scan_add_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_in; if (!enif_inspect_binary(env, argv[0], &bin_in)) return enif_make_badarg(env); int64_t* ptr_in = (int64_t*)bin_in.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[1], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_scan_add_i64(ptr_in, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_slice_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_array_len; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_array_len)) return enif_make_badarg(env);
    size_t val_start; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_start)) return enif_make_badarg(env);
    size_t val_end; if (!enif_get_uint64(env, argv[4], (ErlNifUInt64*)&val_end)) return enif_make_badarg(env);
    size_t res = fp_slice_i64(ptr_input, ptr_output, val_array_len, val_start, val_end);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_sma_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_sma_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_take_n_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_array_len; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_array_len)) return enif_make_badarg(env);
    size_t val_take_count; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_take_count)) return enif_make_badarg(env);
    size_t res = fp_take_n_i64(ptr_input, ptr_output, val_array_len, val_take_count);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_take_while_gt_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    int64_t val_threshold; if (!enif_get_int64(env, argv[3], (ErlNifSInt64*)&val_threshold)) return enif_make_badarg(env);
    size_t res = fp_take_while_gt_i64(ptr_input, ptr_output, val_n, val_threshold);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_training_result_free(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    TrainingResult* res_result; if (!enif_get_resource(env, argv[0], RES_TYPE_TrainingResult, (void**)&res_result)) return enif_make_badarg(env);
    fp_training_result_free(res_result);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_fp_training_result_print(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    TrainingResult* res_result; if (!enif_get_resource(env, argv[0], RES_TYPE_TrainingResult, (void**)&res_result)) return enif_make_badarg(env);
    fp_training_result_print(res_result);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_fp_union_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_array_a; if (!enif_inspect_binary(env, argv[0], &bin_array_a)) return enif_make_badarg(env); int64_t* ptr_array_a = (int64_t*)bin_array_a.data;
    ErlNifBinary bin_array_b; if (!enif_inspect_binary(env, argv[1], &bin_array_b)) return enif_make_badarg(env); int64_t* ptr_array_b = (int64_t*)bin_array_b.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[2], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_len_a; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_len_a)) return enif_make_badarg(env);
    size_t val_len_b; if (!enif_get_uint64(env, argv[4], (ErlNifUInt64*)&val_len_b)) return enif_make_badarg(env);
    size_t res = fp_union_i64(ptr_array_a, ptr_array_b, ptr_output, val_len_a, val_len_b);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_unique_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t res = fp_unique_i64(ptr_input, ptr_output, val_n);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}

static ERL_NIF_TERM nif_fp_wma_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_data; if (!enif_inspect_binary(env, argv[0], &bin_data)) return enif_make_badarg(env); double* ptr_data = (double*)bin_data.data;
    size_t val_n; if (!enif_get_uint64(env, argv[1], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t val_window; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_window)) return enif_make_badarg(env);
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[3], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); double* ptr_output = (double*)out_bin_output.data;
    fp_wma_f64(ptr_data, val_n, val_window, ptr_output);
    return enif_make_binary(env, &out_bin_output);
}

static ERL_NIF_TERM nif_fp_zip_add_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); float* ptr_a = (float*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); float* ptr_b = (float*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); float* ptr_out = (float*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_f32(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_f64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); double* ptr_a = (double*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); double* ptr_b = (double*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); double* ptr_out = (double*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_f64(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_i16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int16_t* ptr_a = (int16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int16_t* ptr_b = (int16_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int16_t* ptr_out = (int16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_i16(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_i32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int32_t* ptr_a = (int32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int32_t* ptr_b = (int32_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int32_t* ptr_out = (int32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_i32(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int64_t* ptr_a = (int64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int64_t* ptr_b = (int64_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int64_t* ptr_out = (int64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_i64(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_i8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); int8_t* ptr_a = (int8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); int8_t* ptr_b = (int8_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); int8_t* ptr_out = (int8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_i8(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_u16(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint16_t* ptr_a = (uint16_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint16_t* ptr_b = (uint16_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint16_t* ptr_out = (uint16_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_u16(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_u32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint32_t* ptr_a = (uint32_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint32_t* ptr_b = (uint32_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint32_t* ptr_out = (uint32_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_u32(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_u64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint64_t* ptr_a = (uint64_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint64_t* ptr_b = (uint64_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint64_t* ptr_out = (uint64_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_u64(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_add_u8(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_a; if (!enif_inspect_binary(env, argv[0], &bin_a)) return enif_make_badarg(env); uint8_t* ptr_a = (uint8_t*)bin_a.data;
    ErlNifBinary bin_b; if (!enif_inspect_binary(env, argv[1], &bin_b)) return enif_make_badarg(env); uint8_t* ptr_b = (uint8_t*)bin_b.data;
    ErlNifUInt64 size_out; if (!enif_get_uint64(env, argv[2], &size_out)) return enif_make_badarg(env); ErlNifBinary out_bin_out; enif_alloc_binary((size_t)size_out, &out_bin_out); uint8_t* ptr_out = (uint8_t*)out_bin_out.data;
    size_t val_n; if (!enif_get_uint64(env, argv[3], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    fp_zip_add_u8(ptr_a, ptr_b, ptr_out, val_n);
    return enif_make_binary(env, &out_bin_out);
}

static ERL_NIF_TERM nif_fp_zip_with_index_i64(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary bin_input; if (!enif_inspect_binary(env, argv[0], &bin_input)) return enif_make_badarg(env); int64_t* ptr_input = (int64_t*)bin_input.data;
    ErlNifUInt64 size_output; if (!enif_get_uint64(env, argv[1], &size_output)) return enif_make_badarg(env); ErlNifBinary out_bin_output; enif_alloc_binary((size_t)size_output, &out_bin_output); int64_t* ptr_output = (int64_t*)out_bin_output.data;
    size_t val_n; if (!enif_get_uint64(env, argv[2], (ErlNifUInt64*)&val_n)) return enif_make_badarg(env);
    size_t res = fp_zip_with_index_i64(ptr_input, ptr_output, val_n);
    return enif_make_tuple2(env, enif_make_uint64(env, res), enif_make_binary(env, &out_bin_output));
}
static ErlNifFunc generated_nif_funcs[] = { {"fp_concat_i64", 5, nif_fp_concat_i64},
    {"fp_contains_i64", 3, nif_fp_contains_i64},
    {"fp_correlation_f64", 3, nif_fp_correlation_f64},
    {"fp_count_i64", 3, nif_fp_count_i64},
    {"fp_covariance_f64", 3, nif_fp_covariance_f64},
    {"fp_detect_outliers_iqr_f64", 4, nif_fp_detect_outliers_iqr_f64},
    {"fp_detect_outliers_zscore_f64", 4, nif_fp_detect_outliers_zscore_f64},
    {"fp_drop_n_i64", 4, nif_fp_drop_n_i64},
    {"fp_drop_while_gt_i64", 4, nif_fp_drop_while_gt_i64},
    {"fp_ema_f64", 4, nif_fp_ema_f64},
    {"fp_filter_gt_i64_simple", 4, nif_fp_filter_gt_i64_simple},
    {"fp_find_index_i64", 3, nif_fp_find_index_i64},
    {"fp_fold_dotp_f32", 3, nif_fp_fold_dotp_f32},
    {"fp_fold_dotp_f64", 3, nif_fp_fold_dotp_f64},
    {"fp_fold_dotp_i16", 3, nif_fp_fold_dotp_i16},
    {"fp_fold_dotp_i32", 3, nif_fp_fold_dotp_i32},
    {"fp_fold_dotp_i64", 3, nif_fp_fold_dotp_i64},
    {"fp_fold_dotp_i8", 3, nif_fp_fold_dotp_i8},
    {"fp_fold_dotp_u16", 3, nif_fp_fold_dotp_u16},
    {"fp_fold_dotp_u32", 3, nif_fp_fold_dotp_u32},
    {"fp_fold_dotp_u64", 3, nif_fp_fold_dotp_u64},
    {"fp_fold_dotp_u8", 3, nif_fp_fold_dotp_u8},
    {"fp_fold_sad_f32", 3, nif_fp_fold_sad_f32},
    {"fp_fold_sad_i16", 3, nif_fp_fold_sad_i16},
    {"fp_fold_sad_i32", 3, nif_fp_fold_sad_i32},
    {"fp_fold_sad_i64", 3, nif_fp_fold_sad_i64},
    {"fp_fold_sad_i8", 3, nif_fp_fold_sad_i8},
    {"fp_fold_sad_u16", 3, nif_fp_fold_sad_u16},
    {"fp_fold_sad_u32", 3, nif_fp_fold_sad_u32},
    {"fp_fold_sad_u64", 3, nif_fp_fold_sad_u64},
    {"fp_fold_sad_u8", 3, nif_fp_fold_sad_u8},
    {"fp_fold_sumsq_f32", 2, nif_fp_fold_sumsq_f32},
    {"fp_fold_sumsq_i16", 2, nif_fp_fold_sumsq_i16},
    {"fp_fold_sumsq_i32", 2, nif_fp_fold_sumsq_i32},
    {"fp_fold_sumsq_i64", 2, nif_fp_fold_sumsq_i64},
    {"fp_fold_sumsq_i8", 2, nif_fp_fold_sumsq_i8},
    {"fp_fold_sumsq_u16", 2, nif_fp_fold_sumsq_u16},
    {"fp_fold_sumsq_u32", 2, nif_fp_fold_sumsq_u32},
    {"fp_fold_sumsq_u64", 2, nif_fp_fold_sumsq_u64},
    {"fp_fold_sumsq_u8", 2, nif_fp_fold_sumsq_u8},
    {"fp_gaussian_nb_predict_batch", 4, nif_fp_gaussian_nb_predict_batch},
    {"fp_gaussian_nb_train", 5, nif_fp_gaussian_nb_train},
    {"fp_group_i64", 4, nif_fp_group_i64},
    {"fp_intersect_i64", 5, nif_fp_intersect_i64},
    {"fp_iterate_add_i64", 4, nif_fp_iterate_add_i64},
    {"fp_iterate_mul_i64", 4, nif_fp_iterate_mul_i64},
    {"fp_kmeans_f64", 7, nif_fp_kmeans_f64},
    {"fp_linear_regression_r2_score", 3, nif_fp_linear_regression_r2_score},
    {"fp_map_abs_f64", 3, nif_fp_map_abs_f64},
    {"fp_map_abs_i64", 3, nif_fp_map_abs_i64},
    {"fp_map_axpy_f32", 5, nif_fp_map_axpy_f32},
    {"fp_map_axpy_f64", 5, nif_fp_map_axpy_f64},
    {"fp_map_axpy_i16", 5, nif_fp_map_axpy_i16},
    {"fp_map_axpy_i32", 5, nif_fp_map_axpy_i32},
    {"fp_map_axpy_i64", 5, nif_fp_map_axpy_i64},
    {"fp_map_axpy_i8", 5, nif_fp_map_axpy_i8},
    {"fp_map_axpy_u16", 5, nif_fp_map_axpy_u16},
    {"fp_map_axpy_u32", 5, nif_fp_map_axpy_u32},
    {"fp_map_axpy_u64", 5, nif_fp_map_axpy_u64},
    {"fp_map_axpy_u8", 5, nif_fp_map_axpy_u8},
    {"fp_map_clamp_f64", 5, nif_fp_map_clamp_f64},
    {"fp_map_clamp_i64", 5, nif_fp_map_clamp_i64},
    {"fp_map_offset_f32", 4, nif_fp_map_offset_f32},
    {"fp_map_offset_f64", 4, nif_fp_map_offset_f64},
    {"fp_map_offset_i16", 4, nif_fp_map_offset_i16},
    {"fp_map_offset_i32", 4, nif_fp_map_offset_i32},
    {"fp_map_offset_i64", 4, nif_fp_map_offset_i64},
    {"fp_map_offset_i8", 4, nif_fp_map_offset_i8},
    {"fp_map_offset_u16", 4, nif_fp_map_offset_u16},
    {"fp_map_offset_u32", 4, nif_fp_map_offset_u32},
    {"fp_map_offset_u64", 4, nif_fp_map_offset_u64},
    {"fp_map_offset_u8", 4, nif_fp_map_offset_u8},
    {"fp_map_scale_f32", 4, nif_fp_map_scale_f32},
    {"fp_map_scale_f64", 4, nif_fp_map_scale_f64},
    {"fp_map_scale_i16", 4, nif_fp_map_scale_i16},
    {"fp_map_scale_i32", 4, nif_fp_map_scale_i32},
    {"fp_map_scale_i64", 4, nif_fp_map_scale_i64},
    {"fp_map_scale_i8", 4, nif_fp_map_scale_i8},
    {"fp_map_scale_u16", 4, nif_fp_map_scale_u16},
    {"fp_map_scale_u32", 4, nif_fp_map_scale_u32},
    {"fp_map_scale_u64", 4, nif_fp_map_scale_u64},
    {"fp_map_scale_u8", 4, nif_fp_map_scale_u8},
    {"fp_map_sqrt_f64", 3, nif_fp_map_sqrt_f64},
    {"fp_moments_f64", 3, nif_fp_moments_f64},
    {"fp_multinomial_nb_predict_batch", 4, nif_fp_multinomial_nb_predict_batch},
    {"fp_multinomial_nb_train", 6, nif_fp_multinomial_nb_train},
    {"fp_neural_network_create", 4, nif_fp_neural_network_create},
    {"fp_neural_network_print_summary", 1, nif_fp_neural_network_print_summary},
    {"fp_neural_network_train", 10, nif_fp_neural_network_train},
    {"fp_partition_gt_i64", 7, nif_fp_partition_gt_i64},
    {"fp_pca_fit", 7, nif_fp_pca_fit},
    {"fp_pca_generate_ellipse_data", 6, nif_fp_pca_generate_ellipse_data},
    {"fp_pca_generate_low_rank_data", 6, nif_fp_pca_generate_low_rank_data},
    {"fp_percentile_f64", 3, nif_fp_percentile_f64},
    {"fp_percentiles_f64", 5, nif_fp_percentiles_f64},
    {"fp_pred_all_eq_const_i64", 3, nif_fp_pred_all_eq_const_i64},
    {"fp_pred_all_gt_zip_i64", 3, nif_fp_pred_all_gt_zip_i64},
    {"fp_pred_any_gt_const_i64", 3, nif_fp_pred_any_gt_const_i64},
    {"fp_range_i64", 3, nif_fp_range_i64},
    {"fp_reduce_add_f32", 2, nif_fp_reduce_add_f32},
    {"fp_reduce_add_f64", 2, nif_fp_reduce_add_f64},
    {"fp_reduce_add_f64_where", 3, nif_fp_reduce_add_f64_where},
    {"fp_reduce_add_i16", 2, nif_fp_reduce_add_i16},
    {"fp_reduce_add_i32", 2, nif_fp_reduce_add_i32},
    {"fp_reduce_add_i64", 2, nif_fp_reduce_add_i64},
    {"fp_reduce_add_i8", 2, nif_fp_reduce_add_i8},
    {"fp_reduce_add_u16", 2, nif_fp_reduce_add_u16},
    {"fp_reduce_add_u32", 2, nif_fp_reduce_add_u32},
    {"fp_reduce_add_u64", 2, nif_fp_reduce_add_u64},
    {"fp_reduce_add_u8", 2, nif_fp_reduce_add_u8},
    {"fp_reduce_and_bool", 2, nif_fp_reduce_and_bool},
    {"fp_reduce_max_f32", 2, nif_fp_reduce_max_f32},
    {"fp_reduce_max_f64", 2, nif_fp_reduce_max_f64},
    {"fp_reduce_max_i16", 2, nif_fp_reduce_max_i16},
    {"fp_reduce_max_i32", 2, nif_fp_reduce_max_i32},
    {"fp_reduce_max_i64", 2, nif_fp_reduce_max_i64},
    {"fp_reduce_max_i8", 2, nif_fp_reduce_max_i8},
    {"fp_reduce_max_u16", 2, nif_fp_reduce_max_u16},
    {"fp_reduce_max_u32", 2, nif_fp_reduce_max_u32},
    {"fp_reduce_max_u64", 2, nif_fp_reduce_max_u64},
    {"fp_reduce_max_u8", 2, nif_fp_reduce_max_u8},
    {"fp_reduce_min_f32", 2, nif_fp_reduce_min_f32},
    {"fp_reduce_min_f64", 2, nif_fp_reduce_min_f64},
    {"fp_reduce_min_i16", 2, nif_fp_reduce_min_i16},
    {"fp_reduce_min_i32", 2, nif_fp_reduce_min_i32},
    {"fp_reduce_min_i64", 2, nif_fp_reduce_min_i64},
    {"fp_reduce_min_i8", 2, nif_fp_reduce_min_i8},
    {"fp_reduce_min_u16", 2, nif_fp_reduce_min_u16},
    {"fp_reduce_min_u32", 2, nif_fp_reduce_min_u32},
    {"fp_reduce_min_u64", 2, nif_fp_reduce_min_u64},
    {"fp_reduce_min_u8", 2, nif_fp_reduce_min_u8},
    {"fp_reduce_mul_f32", 2, nif_fp_reduce_mul_f32},
    {"fp_reduce_mul_i16", 2, nif_fp_reduce_mul_i16},
    {"fp_reduce_mul_i32", 2, nif_fp_reduce_mul_i32},
    {"fp_reduce_mul_i8", 2, nif_fp_reduce_mul_i8},
    {"fp_reduce_mul_u16", 2, nif_fp_reduce_mul_u16},
    {"fp_reduce_mul_u32", 2, nif_fp_reduce_mul_u32},
    {"fp_reduce_mul_u64", 2, nif_fp_reduce_mul_u64},
    {"fp_reduce_mul_u8", 2, nif_fp_reduce_mul_u8},
    {"fp_reduce_or_bool", 2, nif_fp_reduce_or_bool},
    {"fp_reduce_product_f64", 2, nif_fp_reduce_product_f64},
    {"fp_reduce_product_i64", 2, nif_fp_reduce_product_i64},
    {"fp_replicate_f64", 3, nif_fp_replicate_f64},
    {"fp_replicate_i64", 3, nif_fp_replicate_i64},
    {"fp_reverse_i64", 3, nif_fp_reverse_i64},
    {"fp_rolling_max_f64", 4, nif_fp_rolling_max_f64},
    {"fp_rolling_max_i64", 4, nif_fp_rolling_max_i64},
    {"fp_rolling_mean_f64", 4, nif_fp_rolling_mean_f64},
    {"fp_rolling_mean_f64_optimized", 4, nif_fp_rolling_mean_f64_optimized},
    {"fp_rolling_min_f64", 4, nif_fp_rolling_min_f64},
    {"fp_rolling_min_i64", 4, nif_fp_rolling_min_i64},
    {"fp_rolling_range_f64", 4, nif_fp_rolling_range_f64},
    {"fp_rolling_std_f64", 4, nif_fp_rolling_std_f64},
    {"fp_rolling_sum_f64", 4, nif_fp_rolling_sum_f64},
    {"fp_rolling_sum_f64_optimized", 4, nif_fp_rolling_sum_f64_optimized},
    {"fp_rolling_sum_i64", 4, nif_fp_rolling_sum_i64},
    {"fp_rolling_variance_f64", 4, nif_fp_rolling_variance_f64},
    {"fp_run_length_encode_i64", 3, nif_fp_run_length_encode_i64},
    {"fp_scan_add_f64", 3, nif_fp_scan_add_f64},
    {"fp_scan_add_i64", 3, nif_fp_scan_add_i64},
    {"fp_slice_i64", 5, nif_fp_slice_i64},
    {"fp_sma_f64", 4, nif_fp_sma_f64},
    {"fp_take_n_i64", 4, nif_fp_take_n_i64},
    {"fp_take_while_gt_i64", 4, nif_fp_take_while_gt_i64},
    {"fp_training_result_free", 1, nif_fp_training_result_free},
    {"fp_training_result_print", 1, nif_fp_training_result_print},
    {"fp_union_i64", 5, nif_fp_union_i64},
    {"fp_unique_i64", 3, nif_fp_unique_i64},
    {"fp_wma_f64", 4, nif_fp_wma_f64},
    {"fp_zip_add_f32", 4, nif_fp_zip_add_f32},
    {"fp_zip_add_f64", 4, nif_fp_zip_add_f64},
    {"fp_zip_add_i16", 4, nif_fp_zip_add_i16},
    {"fp_zip_add_i32", 4, nif_fp_zip_add_i32},
    {"fp_zip_add_i64", 4, nif_fp_zip_add_i64},
    {"fp_zip_add_i8", 4, nif_fp_zip_add_i8},
    {"fp_zip_add_u16", 4, nif_fp_zip_add_u16},
    {"fp_zip_add_u32", 4, nif_fp_zip_add_u32},
    {"fp_zip_add_u64", 4, nif_fp_zip_add_u64},
    {"fp_zip_add_u8", 4, nif_fp_zip_add_u8},
    {"fp_zip_with_index_i64", 3, nif_fp_zip_with_index_i64} };
static int load_resources(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) { RES_TYPE_GaussianNBModel = enif_open_resource_type(env, NULL, "GaussianNBModel", dtor_GaussianNBModel, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_KMeansResult = enif_open_resource_type(env, NULL, "KMeansResult", dtor_KMeansResult, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_MultinomialNBModel = enif_open_resource_type(env, NULL, "MultinomialNBModel", dtor_MultinomialNBModel, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_NeuralNetwork = enif_open_resource_type(env, NULL, "NeuralNetwork", dtor_NeuralNetwork, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_PCAModel = enif_open_resource_type(env, NULL, "PCAModel", dtor_PCAModel, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_PCAResult = enif_open_resource_type(env, NULL, "PCAResult", dtor_PCAResult, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    RES_TYPE_TrainingResult = enif_open_resource_type(env, NULL, "TrainingResult", dtor_TrainingResult, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
return 0; }