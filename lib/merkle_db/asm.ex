defmodule MerkleDb.ASM do
  @on_load :load_nifs
  def load_nifs do
    priv_dir =
      case :code.priv_dir(:merkle_db) do
        {:error, _} ->
          Path.expand("../../priv", __DIR__)
        dir ->
          List.to_string(dir)
      end

    base_path = Path.join(priv_dir, "merkle_nif")
    path =
      if nif_exists?(base_path) do
        base_path
      else
        Path.join(Path.expand("../../priv", __DIR__), "merkle_nif")
      end

    case :erlang.load_nif(String.to_charlist(path), 0) do
      :ok -> :ok
      {:error, {:already_loaded, _}} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp nif_exists?(base_path) do
    Enum.any?([".dll", ".so", ".dylib"], fn ext ->
      File.exists?(base_path <> ext)
    end)
  end
@doc "Calls C function: fp_concat_i64"
def fp_concat_i64(_input_a, _input_b, _size_output, _len_a, _len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_contains_i64"
def fp_contains_i64(_input, _n, _target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_correlation_f64"
def fp_correlation_f64(_x, _y, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_count_i64"
def fp_count_i64(_input, _n, _target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_covariance_f64"
def fp_covariance_f64(_x, _y, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_detect_outliers_iqr_f64"
def fp_detect_outliers_iqr_f64(_data, _n, _k_factor, _size_is_outlier), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_detect_outliers_zscore_f64"
def fp_detect_outliers_zscore_f64(_data, _n, _threshold, _size_is_outlier), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_drop_n_i64"
def fp_drop_n_i64(_input, _size_output, _array_len, _drop_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_drop_while_gt_i64"
def fp_drop_while_gt_i64(_input, _size_output, _n, _threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_ema_f64"
def fp_ema_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_filter_gt_i64_simple"
def fp_filter_gt_i64_simple(_input, _size_output, _n, _threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_find_index_i64"
def fp_find_index_i64(_input, _n, _target), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_f32"
def fp_fold_dotp_f32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_f64"
def fp_fold_dotp_f64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i16"
def fp_fold_dotp_i16(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i32"
def fp_fold_dotp_i32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i64"
def fp_fold_dotp_i64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_i8"
def fp_fold_dotp_i8(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u16"
def fp_fold_dotp_u16(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u32"
def fp_fold_dotp_u32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u64"
def fp_fold_dotp_u64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_dotp_u8"
def fp_fold_dotp_u8(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_f32"
def fp_fold_sad_f32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i16"
def fp_fold_sad_i16(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i32"
def fp_fold_sad_i32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i64"
def fp_fold_sad_i64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_i8"
def fp_fold_sad_i8(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u16"
def fp_fold_sad_u16(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u32"
def fp_fold_sad_u32(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u64"
def fp_fold_sad_u64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sad_u8"
def fp_fold_sad_u8(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_f32"
def fp_fold_sumsq_f32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i16"
def fp_fold_sumsq_i16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i32"
def fp_fold_sumsq_i32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i64"
def fp_fold_sumsq_i64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_i8"
def fp_fold_sumsq_i8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u16"
def fp_fold_sumsq_u16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u32"
def fp_fold_sumsq_u32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u64"
def fp_fold_sumsq_u64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_fold_sumsq_u8"
def fp_fold_sumsq_u8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_gaussian_nb_predict_batch"
def fp_gaussian_nb_predict_batch(_model, _X, _n, _size_predictions), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_gaussian_nb_train"
def fp_gaussian_nb_train(_X, _y, _n, _d, _n_classes), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_group_i64"
def fp_group_i64(_input, _size_groups_out, _size_counts_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_intersect_i64"
def fp_intersect_i64(_array_a, _array_b, _size_output, _len_a, _len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_iterate_add_i64"
def fp_iterate_add_i64(_size_output, _n, _start, _step), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_iterate_mul_i64"
def fp_iterate_mul_i64(_size_output, _n, _start, _factor), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_kmeans_f64"
def fp_kmeans_f64(_data, _n, _d, _k, _max_iter, _tol, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_linear_regression_r2_score"
def fp_linear_regression_r2_score(_y_true, _y_pred, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_abs_f64"
def fp_map_abs_f64(_in_, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_abs_i64"
def fp_map_abs_i64(_in_, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_f32"
def fp_map_axpy_f32(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_f64"
def fp_map_axpy_f64(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i16"
def fp_map_axpy_i16(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i32"
def fp_map_axpy_i32(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i64"
def fp_map_axpy_i64(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_i8"
def fp_map_axpy_i8(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u16"
def fp_map_axpy_u16(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u32"
def fp_map_axpy_u32(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u64"
def fp_map_axpy_u64(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_axpy_u8"
def fp_map_axpy_u8(_x, _y, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_clamp_f64"
def fp_map_clamp_f64(_in_, _size_out, _n, _min_val, _max_val), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_clamp_i64"
def fp_map_clamp_i64(_in_, _size_out, _n, _min_val, _max_val), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_f32"
def fp_map_offset_f32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_f64"
def fp_map_offset_f64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i16"
def fp_map_offset_i16(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i32"
def fp_map_offset_i32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i64"
def fp_map_offset_i64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_i8"
def fp_map_offset_i8(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u16"
def fp_map_offset_u16(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u32"
def fp_map_offset_u32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u64"
def fp_map_offset_u64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_offset_u8"
def fp_map_offset_u8(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_f32"
def fp_map_scale_f32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_f64"
def fp_map_scale_f64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i16"
def fp_map_scale_i16(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i32"
def fp_map_scale_i32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i64"
def fp_map_scale_i64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_i8"
def fp_map_scale_i8(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u16"
def fp_map_scale_u16(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u32"
def fp_map_scale_u32(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u64"
def fp_map_scale_u64(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_scale_u8"
def fp_map_scale_u8(_in_, _size_out, _n, _c), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_map_sqrt_f64"
def fp_map_sqrt_f64(_in_, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_moments_f64"
def fp_moments_f64(_data, _n, _size_moments), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_multinomial_nb_predict_batch"
def fp_multinomial_nb_predict_batch(_model, _X, _n, _size_predictions), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_multinomial_nb_train"
def fp_multinomial_nb_train(_X, _y, _n, _d, _n_classes, _alpha), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_create"
def fp_neural_network_create(_n_inputs, _n_hidden, _n_outputs, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_print_summary"
def fp_neural_network_print_summary(_net), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_neural_network_train"
def fp_neural_network_train(_n_inputs, _n_hidden, _n_outputs, _X_train, _y_train, _n_samples, _n_epochs, _learning_rate, _verbose, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_partition_gt_i64"
def fp_partition_gt_i64(_input, _size_output_pass, _size_output_fail, _n, _threshold, _size_out_pass_count, _size_out_fail_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_fit"
def fp_pca_fit(_X, _n, _d, _n_components, _max_iterations, _tolerance, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_generate_ellipse_data"
def fp_pca_generate_ellipse_data(_size_X, _n, _major_axis, _minor_axis, _angle, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pca_generate_low_rank_data"
def fp_pca_generate_low_rank_data(_size_X, _n, _d, _intrinsic_dim, _noise_stddev, _seed), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_percentile_f64"
def fp_percentile_f64(_data, _n, _p), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_percentiles_f64"
def fp_percentiles_f64(_data, _n, _p_values, _n_percentiles, _size_results), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_all_eq_const_i64"
def fp_pred_all_eq_const_i64(_arr, _n, _value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_all_gt_zip_i64"
def fp_pred_all_gt_zip_i64(_a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_pred_any_gt_const_i64"
def fp_pred_any_gt_const_i64(_arr, _n, _value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_range_i64"
def fp_range_i64(_size_output, _start, _end_), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f32"
def fp_reduce_add_f32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f64"
def fp_reduce_add_f64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_f64_where"
def fp_reduce_add_f64_where(_x, _mask, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i16"
def fp_reduce_add_i16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i32"
def fp_reduce_add_i32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i64"
def fp_reduce_add_i64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_i8"
def fp_reduce_add_i8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u16"
def fp_reduce_add_u16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u32"
def fp_reduce_add_u32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u64"
def fp_reduce_add_u64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_add_u8"
def fp_reduce_add_u8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_and_bool"
def fp_reduce_and_bool(_input, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_f32"
def fp_reduce_max_f32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_f64"
def fp_reduce_max_f64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i16"
def fp_reduce_max_i16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i32"
def fp_reduce_max_i32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i64"
def fp_reduce_max_i64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_i8"
def fp_reduce_max_i8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u16"
def fp_reduce_max_u16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u32"
def fp_reduce_max_u32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u64"
def fp_reduce_max_u64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_max_u8"
def fp_reduce_max_u8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_f32"
def fp_reduce_min_f32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_f64"
def fp_reduce_min_f64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i16"
def fp_reduce_min_i16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i32"
def fp_reduce_min_i32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i64"
def fp_reduce_min_i64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_i8"
def fp_reduce_min_i8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u16"
def fp_reduce_min_u16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u32"
def fp_reduce_min_u32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u64"
def fp_reduce_min_u64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_min_u8"
def fp_reduce_min_u8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_f32"
def fp_reduce_mul_f32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i16"
def fp_reduce_mul_i16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i32"
def fp_reduce_mul_i32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_i8"
def fp_reduce_mul_i8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u16"
def fp_reduce_mul_u16(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u32"
def fp_reduce_mul_u32(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u64"
def fp_reduce_mul_u64(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_mul_u8"
def fp_reduce_mul_u8(_in_, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_or_bool"
def fp_reduce_or_bool(_input, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_product_f64"
def fp_reduce_product_f64(_input, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reduce_product_i64"
def fp_reduce_product_i64(_input, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_replicate_f64"
def fp_replicate_f64(_size_output, _n, _value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_replicate_i64"
def fp_replicate_i64(_size_output, _n, _value), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_reverse_i64"
def fp_reverse_i64(_input, _size_output, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_max_f64"
def fp_rolling_max_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_max_i64"
def fp_rolling_max_i64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_mean_f64"
def fp_rolling_mean_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_mean_f64_optimized"
def fp_rolling_mean_f64_optimized(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_min_f64"
def fp_rolling_min_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_min_i64"
def fp_rolling_min_i64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_range_f64"
def fp_rolling_range_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_std_f64"
def fp_rolling_std_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_f64"
def fp_rolling_sum_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_f64_optimized"
def fp_rolling_sum_f64_optimized(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_sum_i64"
def fp_rolling_sum_i64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_rolling_variance_f64"
def fp_rolling_variance_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_run_length_encode_i64"
def fp_run_length_encode_i64(_input, _size_output, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_scan_add_f64"
def fp_scan_add_f64(_in_, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_scan_add_i64"
def fp_scan_add_i64(_in_, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_slice_i64"
def fp_slice_i64(_input, _size_output, _array_len, _start, _end_), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_sma_f64"
def fp_sma_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_take_n_i64"
def fp_take_n_i64(_input, _size_output, _array_len, _take_count), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_take_while_gt_i64"
def fp_take_while_gt_i64(_input, _size_output, _n, _threshold), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_training_result_free"
def fp_training_result_free(_result), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_training_result_print"
def fp_training_result_print(_result), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_union_i64"
def fp_union_i64(_array_a, _array_b, _size_output, _len_a, _len_b), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_unique_i64"
def fp_unique_i64(_input, _size_output, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_wma_f64"
def fp_wma_f64(_data, _n, _window, _size_output), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_f32"
def fp_zip_add_f32(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_f64"
def fp_zip_add_f64(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i16"
def fp_zip_add_i16(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i32"
def fp_zip_add_i32(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i64"
def fp_zip_add_i64(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_i8"
def fp_zip_add_i8(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u16"
def fp_zip_add_u16(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u32"
def fp_zip_add_u32(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u64"
def fp_zip_add_u64(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_add_u8"
def fp_zip_add_u8(_a, _b, _size_out, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_zip_with_index_i64"
def fp_zip_with_index_i64(_input, _size_output, _n), do: :erlang.nif_error(:nif_not_loaded)

@doc "Calls C function: fp_job_start"
def fp_job_start(_op, _args, _opts), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_job_status"
def fp_job_status(_job), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_job_result"
def fp_job_result(_job), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_job_cancel"
def fp_job_cancel(_job), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_query_gemv_columnar_batch"
def fp_query_gemv_columnar_batch(_columns_tuple, _queries_bin, _batch_count, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_query_gemv_f32_batch"
def fp_query_gemv_f32_batch(_db_vectors_bin, _query_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_vector_sum_f32"
def fp_vector_sum_f32(_vectors_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_quantize_f64_to_u8"
def fp_quantize_f64_to_u8(_in_bin, _min_val, _inv_scale), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_quantize_f32_to_u8"
def fp_quantize_f32_to_u8(_in_bin, _min_val, _inv_scale), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_query_gemv_quantized"
def fp_query_gemv_quantized(_columns_tuple, _query_bin, _bias, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_query_gemv_quantized_f32"
def fp_query_gemv_quantized_f32(_columns_tuple, _query_bin, _bias, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_sparse_dotp"
def fp_sparse_dotp(_indices_a, _values_a, _indices_b, _values_b), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_mmap_open"
def fp_mmap_open(_path), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_query_gemv_mmap_f32"
def fp_query_gemv_mmap_f32(_mmap_resources_tuple, _query_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_hnsw_create"
def fp_hnsw_create(_dim, _m, _ef_construction, _capacity), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_hnsw_insert"
def fp_hnsw_insert(_hnsw_res, _vector_idx, _vec_bin, _columns_tuple, _count), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_hnsw_search"
def fp_hnsw_search(_hnsw_res, _query_bin, _k, _ef_search, _columns_tuple, _count), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_pca_transform_result"
def fp_pca_transform_result(_pca_result, _data_bin, _count), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_pca_result_n_components"
def fp_pca_result_n_components(_pca_result), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_pca_result_total_variance"
def fp_pca_result_total_variance(_pca_result), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_pca_result_explained_variance"
def fp_pca_result_explained_variance(_pca_result, _size), do: :erlang.nif_error(:nif_not_loaded)
@doc "Calls C function: fp_pca_result_cumulative_variance"
def fp_pca_result_cumulative_variance(_pca_result, _size), do: :erlang.nif_error(:nif_not_loaded)

# --- Struct Accessors ---
def get_KMeansResult_centroids(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_assignments(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_inertia(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_KMeansResult_converged(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_n_components(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_eigenvalues(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAModel_total_variance(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
def get_PCAResult_converged(_res, _size \\ 0), do: :erlang.nif_error(:nif_not_loaded)
# --- Query Kernels (manually added) ---

@doc """
Single-pass columnar GEMV for vector similarity search.
Replaces the 64-allocation loop with one NIF call.

columns_tuple: tuple of dim binaries (each count*8 bytes)
query_bin: normalized query vector (dim*8 bytes)
count: number of vectors
dim: dimension

Returns: scores binary (count*8 bytes)
"""
def fp_query_gemv_columnar(_columns_tuple, _query_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)

@doc """
Indexed columnar GEMV for IVF search.
Only computes scores for specified row indices - O(num_indices * dim) instead of O(count * dim).

columns_tuple: tuple of dim column binaries
query_bin: normalized query vector (dim*8 bytes)
indices_bin: int32 array of candidate row indices
count: total number of vectors
dim: vector dimension

Returns: scores binary (num_indices*8 bytes, in same order as indices)
"""
def fp_query_gemv_indexed(_columns_tuple, _query_bin, _indices_bin, _count, _dim), do: :erlang.nif_error(:nif_not_loaded)

@doc """
Top-K selection from scores array.

scores_bin: scores binary (count*8 bytes)
count: number of scores
k: number of top results
threshold: minimum score threshold

Returns: {result_count, indices_bin, scores_bin}
  - indices_bin: int32 indices of top results
  - scores_bin: float64 scores of top results
"""
def fp_query_topk(_scores_bin, _count, _k, _threshold), do: :erlang.nif_error(:nif_not_loaded)

# --- BLAKE3 Cryptographic Hash ---

@doc """
BLAKE3 cryptographic hash (3-5x faster than SHA-256).

input: binary data to hash

Returns: 32-byte hash binary
"""
def fp_blake3_hash(_input), do: :erlang.nif_error(:nif_not_loaded)

end