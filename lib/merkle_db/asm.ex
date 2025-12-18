defmodule MerkleDb.ASM do
  @on_load :load_nifs

  def load_nifs do
    path = :code.priv_dir(:merkle_db) ++ ~c"/merkle_nif"
    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, {reason, text}} -> 
        IO.puts("\n🔴 CRITICAL: NIF Load Failed: #{inspect(reason)} - #{text}")
        {:error, reason}
    end
  end

  # ===========================================================================
  #  FULL NIF INTERFACE (125 Functions)
  # ===========================================================================

  # --- Core ---
  def fp_concat_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_contains_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_count_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_find_index_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_group_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_intersect_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_partition_gt_i64(_a, _b, _c, _d, _e, _f, _g), do: :erlang.nif_error(:nif_not_loaded)
  def fp_range_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_replicate_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_replicate_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reverse_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_run_length_encode_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_slice_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_take_n_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_take_while_gt_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_union_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_unique_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  # --- Statistics ---
  def fp_correlation_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_covariance_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_detect_outliers_iqr_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_detect_outliers_zscore_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_ema_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_linear_regression_r2_score(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_moments_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_percentile_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_percentiles_f64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_sma_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_wma_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)

  # --- Filters ---
  def fp_drop_n_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_drop_while_gt_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_filter_gt_i64_simple(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pred_all_eq_const_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pred_all_gt_zip_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pred_any_gt_const_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  # --- ML Models ---
  def fp_gaussian_nb_predict_batch(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_gaussian_nb_train(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_kmeans_f64(_a, _b, _c, _d, _e, _f, _g), do: :erlang.nif_error(:nif_not_loaded)
  def fp_multinomial_nb_predict_batch(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_multinomial_nb_train(_a, _b, _c, _d, _e, _f), do: :erlang.nif_error(:nif_not_loaded)
  def fp_neural_network_create(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_neural_network_print_summary(_a), do: :erlang.nif_error(:nif_not_loaded)
  def fp_neural_network_train(_a, _b, _c, _d, _e, _f, _g, _h, _i, _j), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pca_fit(_a, _b, _c, _d, _e, _f, _g), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pca_generate_ellipse_data(_a, _b, _c, _d, _e, _f), do: :erlang.nif_error(:nif_not_loaded)
  def fp_pca_generate_low_rank_data(_a, _b, _c, _d, _e, _f), do: :erlang.nif_error(:nif_not_loaded)
  def fp_training_result_free(_a), do: :erlang.nif_error(:nif_not_loaded)
  def fp_training_result_print(_a), do: :erlang.nif_error(:nif_not_loaded)

  # --- Iteration ---
  def fp_iterate_add_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_iterate_mul_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)

  # --- Map (Element-wise) ---
  def fp_map_abs_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_abs_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_clamp_f64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_clamp_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_sqrt_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  # AXPY (The workhorse)
  def fp_map_axpy_f32(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_f64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_i16(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_i32(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_i64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_i8(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_u16(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_u32(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_u64(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_axpy_u8(_a, _b, _c, _d, _e), do: :erlang.nif_error(:nif_not_loaded)

  # Map Offset / Scale
  def fp_map_offset_f32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_i16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_i32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_i8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_u16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_u32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_u64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_offset_u8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)

  def fp_map_scale_f32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_i16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_i32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_i8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_u16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_u32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_u64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_map_scale_u8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)

  # --- Fold / Reduction ---
  def fp_fold_dotp_f32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_i16(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_i32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_i8(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_u16(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_u32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_u64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_dotp_u8(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  def fp_fold_sad_f32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_i16(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_i32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_i8(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_u16(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_u32(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_u64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sad_u8(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  def fp_fold_sumsq_f32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_i16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_i32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_i64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_i8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_u16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_u32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_u64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_fold_sumsq_u8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_add_f32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_f64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_f64_where(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_i16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_i32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_i64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_i8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_u16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_u32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_u64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_add_u8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_and_bool(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_or_bool(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_max_f32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_f64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_i16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_i32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_i64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_i8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_u16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_u32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_u64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_max_u8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_min_f32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_f64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_i16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_i32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_i64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_i8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_u16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_u32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_u64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_min_u8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_mul_f32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_i16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_i32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_i8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_u16(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_u32(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_u64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_mul_u8(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def fp_reduce_product_f64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def fp_reduce_product_i64(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  # --- Rolling ---
  def fp_rolling_max_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_max_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_mean_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_mean_f64_optimized(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_min_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_min_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_range_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_std_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_sum_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_sum_f64_optimized(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_sum_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_rolling_variance_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)

  # --- Scan ---
  def fp_scan_add_f64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
  def fp_scan_add_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)

  # --- Zip ---
  def fp_zip_add_f32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_f64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_i16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_i32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_i64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_i8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_u16(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_u32(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_u64(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_add_u8(_a, _b, _c, _d), do: :erlang.nif_error(:nif_not_loaded)
  def fp_zip_with_index_i64(_a, _b, _c), do: :erlang.nif_error(:nif_not_loaded)
end