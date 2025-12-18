defmodule MerkleDb.ASM do
  @on_load :load_nifs
  def load_nifs do
    path = :code.priv_dir(:merkle_db) |> Path.join("merkle_nif") |> String.to_charlist()
    :erlang.load_nif(path, 0)
  end
@doc "Calls C function: fp_concat_i64"
def fp_concat_i64(input_a, input_b, size_output, len_a, len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_contains_i64"
def fp_contains_i64(input, n, target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_correlation_f64"
def fp_correlation_f64(x, y, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_count_i64"
def fp_count_i64(input, n, target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_covariance_f64"
def fp_covariance_f64(x, y, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_detect_outliers_iqr_f64"
def fp_detect_outliers_iqr_f64(data, n, k_factor, size_is_outlier), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_detect_outliers_zscore_f64"
def fp_detect_outliers_zscore_f64(data, n, threshold, size_is_outlier), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_drop_n_i64"
def fp_drop_n_i64(input, size_output, array_len, drop_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_drop_while_gt_i64"
def fp_drop_while_gt_i64(input, size_output, n, threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_ema_f64"
def fp_ema_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_filter_gt_i64_simple"
def fp_filter_gt_i64_simple(input, size_output, n, threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_find_index_i64"
def fp_find_index_i64(input, n, target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_f32"
def fp_fold_dotp_f32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_f64"
def fp_fold_dotp_f64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i16"
def fp_fold_dotp_i16(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i32"
def fp_fold_dotp_i32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i64"
def fp_fold_dotp_i64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i8"
def fp_fold_dotp_i8(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u16"
def fp_fold_dotp_u16(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u32"
def fp_fold_dotp_u32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u64"
def fp_fold_dotp_u64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u8"
def fp_fold_dotp_u8(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_f32"
def fp_fold_sad_f32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i16"
def fp_fold_sad_i16(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i32"
def fp_fold_sad_i32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i64"
def fp_fold_sad_i64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i8"
def fp_fold_sad_i8(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u16"
def fp_fold_sad_u16(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u32"
def fp_fold_sad_u32(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u64"
def fp_fold_sad_u64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u8"
def fp_fold_sad_u8(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_f32"
def fp_fold_sumsq_f32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i16"
def fp_fold_sumsq_i16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i32"
def fp_fold_sumsq_i32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i64"
def fp_fold_sumsq_i64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i8"
def fp_fold_sumsq_i8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u16"
def fp_fold_sumsq_u16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u32"
def fp_fold_sumsq_u32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u64"
def fp_fold_sumsq_u64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u8"
def fp_fold_sumsq_u8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_gaussian_nb_predict_batch"
def fp_gaussian_nb_predict_batch(model, X, n, size_predictions), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_gaussian_nb_train"
def fp_gaussian_nb_train(X, y, n, d, n_classes), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_group_i64"
def fp_group_i64(input, size_groups_out, size_counts_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_intersect_i64"
def fp_intersect_i64(array_a, array_b, size_output, len_a, len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_iterate_add_i64"
def fp_iterate_add_i64(size_output, n, start, step), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_iterate_mul_i64"
def fp_iterate_mul_i64(size_output, n, start, factor), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_kmeans_f64"
def fp_kmeans_f64(data, n, d, k, max_iter, tol, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_linear_regression_r2_score"
def fp_linear_regression_r2_score(y_true, y_pred, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_abs_f64"
def fp_map_abs_f64(in_, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_abs_i64"
def fp_map_abs_i64(in_, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_f32"
def fp_map_axpy_f32(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_f64"
def fp_map_axpy_f64(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i16"
def fp_map_axpy_i16(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i32"
def fp_map_axpy_i32(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i64"
def fp_map_axpy_i64(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i8"
def fp_map_axpy_i8(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u16"
def fp_map_axpy_u16(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u32"
def fp_map_axpy_u32(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u64"
def fp_map_axpy_u64(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u8"
def fp_map_axpy_u8(x, y, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_clamp_f64"
def fp_map_clamp_f64(in_, size_out, n, min_val, max_val), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_clamp_i64"
def fp_map_clamp_i64(in_, size_out, n, min_val, max_val), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_f32"
def fp_map_offset_f32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_f64"
def fp_map_offset_f64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i16"
def fp_map_offset_i16(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i32"
def fp_map_offset_i32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i64"
def fp_map_offset_i64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i8"
def fp_map_offset_i8(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u16"
def fp_map_offset_u16(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u32"
def fp_map_offset_u32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u64"
def fp_map_offset_u64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u8"
def fp_map_offset_u8(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_f32"
def fp_map_scale_f32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_f64"
def fp_map_scale_f64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i16"
def fp_map_scale_i16(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i32"
def fp_map_scale_i32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i64"
def fp_map_scale_i64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i8"
def fp_map_scale_i8(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u16"
def fp_map_scale_u16(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u32"
def fp_map_scale_u32(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u64"
def fp_map_scale_u64(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u8"
def fp_map_scale_u8(in_, size_out, n, c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_sqrt_f64"
def fp_map_sqrt_f64(in_, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_moments_f64"
def fp_moments_f64(data, n, size_moments), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_multinomial_nb_predict_batch"
def fp_multinomial_nb_predict_batch(model, X, n, size_predictions), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_multinomial_nb_train"
def fp_multinomial_nb_train(X, y, n, d, n_classes, alpha), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_create"
def fp_neural_network_create(n_inputs, n_hidden, n_outputs, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_print_summary"
def fp_neural_network_print_summary(net), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_train"
def fp_neural_network_train(n_inputs, n_hidden, n_outputs, X_train, y_train, n_samples, n_epochs, learning_rate, verbose, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_partition_gt_i64"
def fp_partition_gt_i64(input, size_output_pass, size_output_fail, n, threshold, size_out_pass_count, size_out_fail_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_fit"
def fp_pca_fit(X, n, d, n_components, max_iterations, tolerance, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_generate_ellipse_data"
def fp_pca_generate_ellipse_data(size_X, n, major_axis, minor_axis, angle, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_generate_low_rank_data"
def fp_pca_generate_low_rank_data(size_X, n, d, intrinsic_dim, noise_stddev, seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_percentile_f64"
def fp_percentile_f64(data, n, p), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_percentiles_f64"
def fp_percentiles_f64(data, n, p_values, n_percentiles, size_results), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_all_eq_const_i64"
def fp_pred_all_eq_const_i64(arr, n, value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_all_gt_zip_i64"
def fp_pred_all_gt_zip_i64(a, b, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_any_gt_const_i64"
def fp_pred_any_gt_const_i64(arr, n, value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_range_i64"
def fp_range_i64(size_output, start, end_), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f32"
def fp_reduce_add_f32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f64"
def fp_reduce_add_f64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f64_where"
def fp_reduce_add_f64_where(x, mask, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i16"
def fp_reduce_add_i16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i32"
def fp_reduce_add_i32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i64"
def fp_reduce_add_i64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i8"
def fp_reduce_add_i8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u16"
def fp_reduce_add_u16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u32"
def fp_reduce_add_u32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u64"
def fp_reduce_add_u64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u8"
def fp_reduce_add_u8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_and_bool"
def fp_reduce_and_bool(input, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_f32"
def fp_reduce_max_f32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_f64"
def fp_reduce_max_f64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i16"
def fp_reduce_max_i16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i32"
def fp_reduce_max_i32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i64"
def fp_reduce_max_i64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i8"
def fp_reduce_max_i8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u16"
def fp_reduce_max_u16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u32"
def fp_reduce_max_u32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u64"
def fp_reduce_max_u64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u8"
def fp_reduce_max_u8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_f32"
def fp_reduce_min_f32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_f64"
def fp_reduce_min_f64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i16"
def fp_reduce_min_i16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i32"
def fp_reduce_min_i32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i64"
def fp_reduce_min_i64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i8"
def fp_reduce_min_i8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u16"
def fp_reduce_min_u16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u32"
def fp_reduce_min_u32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u64"
def fp_reduce_min_u64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u8"
def fp_reduce_min_u8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_f32"
def fp_reduce_mul_f32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i16"
def fp_reduce_mul_i16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i32"
def fp_reduce_mul_i32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i8"
def fp_reduce_mul_i8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u16"
def fp_reduce_mul_u16(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u32"
def fp_reduce_mul_u32(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u64"
def fp_reduce_mul_u64(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u8"
def fp_reduce_mul_u8(in_, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_or_bool"
def fp_reduce_or_bool(input, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_product_f64"
def fp_reduce_product_f64(input, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_product_i64"
def fp_reduce_product_i64(input, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_replicate_f64"
def fp_replicate_f64(size_output, n, value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_replicate_i64"
def fp_replicate_i64(size_output, n, value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reverse_i64"
def fp_reverse_i64(input, size_output, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_max_f64"
def fp_rolling_max_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_max_i64"
def fp_rolling_max_i64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_mean_f64"
def fp_rolling_mean_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_mean_f64_optimized"
def fp_rolling_mean_f64_optimized(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_min_f64"
def fp_rolling_min_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_min_i64"
def fp_rolling_min_i64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_range_f64"
def fp_rolling_range_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_std_f64"
def fp_rolling_std_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_f64"
def fp_rolling_sum_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_f64_optimized"
def fp_rolling_sum_f64_optimized(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_i64"
def fp_rolling_sum_i64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_variance_f64"
def fp_rolling_variance_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_run_length_encode_i64"
def fp_run_length_encode_i64(input, size_output, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_scan_add_f64"
def fp_scan_add_f64(in_, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_scan_add_i64"
def fp_scan_add_i64(in_, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_slice_i64"
def fp_slice_i64(input, size_output, array_len, start, end_), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_sma_f64"
def fp_sma_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_take_n_i64"
def fp_take_n_i64(input, size_output, array_len, take_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_take_while_gt_i64"
def fp_take_while_gt_i64(input, size_output, n, threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_training_result_free"
def fp_training_result_free(result), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_training_result_print"
def fp_training_result_print(result), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_union_i64"
def fp_union_i64(array_a, array_b, size_output, len_a, len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_unique_i64"
def fp_unique_i64(input, size_output, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_wma_f64"
def fp_wma_f64(data, n, window, size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_f32"
def fp_zip_add_f32(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_f64"
def fp_zip_add_f64(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i16"
def fp_zip_add_i16(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i32"
def fp_zip_add_i32(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i64"
def fp_zip_add_i64(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i8"
def fp_zip_add_i8(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u16"
def fp_zip_add_u16(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u32"
def fp_zip_add_u32(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u64"
def fp_zip_add_u64(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u8"
def fp_zip_add_u8(a, b, size_out, n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_with_index_i64"
def fp_zip_with_index_i64(input, size_output, n), do: :erlang.nif_error(:nif_not_loaded)

# --- Struct Accessors ---
def get_KMeansResult_centroids(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_assignments(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_inertia(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_converged(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_n_components(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_eigenvalues(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_total_variance(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAResult_converged(res, size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
end
