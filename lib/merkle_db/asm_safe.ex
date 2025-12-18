defmodule MerkleDb.ASM.Safe do
  @moduledoc """
  Safe wrapper around MerkleDb.ASM with:
  - Input validation
  - Timeout protection
  - {:ok, result} | {:error, reason} tuple returns
  - Better error messages
  """

  alias MerkleDb.ASM

  @default_timeout 30_000  # 30 seconds

  # ==================== K-Means ====================

  @doc """
  Safe K-Means clustering with validation and timeout.

  Returns: {:ok, result} | {:error, reason}
  """
  def fp_kmeans_f64(data, n, d, k, max_iter, tol, seed, timeout \\ @default_timeout) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_positive("d", d),
         :ok <- validate_positive("k", k),
         :ok <- validate_k_vs_n(k, n),
         :ok <- validate_positive("max_iter", max_iter),
         :ok <- validate_tolerance(tol),
         :ok <- validate_binary_size(data, n * d * 8, "data") do

      execute_with_timeout(fn ->
        result = ASM.fp_kmeans_f64(data, n, d, k, max_iter, tol, seed)
        {:ok, result}
      end, timeout)
    end
  end

  # ==================== PCA ====================

  @doc """
  Safe PCA with validation and timeout.
  """
  def fp_pca_fit(X, n, d, n_components, max_iterations, tolerance, seed, timeout \\ @default_timeout) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_positive("d", d),
         :ok <- validate_positive("n_components", n_components),
         :ok <- validate_range(n_components, 1, d, "n_components must be <= d"),
         :ok <- validate_positive("max_iterations", max_iterations),
         :ok <- validate_tolerance(tolerance),
         :ok <- validate_binary_size(X, n * d * 8, "X") do

      execute_with_timeout(fn ->
        result = ASM.fp_pca_fit(X, n, d, n_components, max_iterations, tolerance, seed)
        {:ok, result}
      end, timeout)
    end
  end

  # ==================== Neural Networks ====================

  @doc """
  Safe neural network training with validation and timeout.
  """
  def fp_neural_network_train(n_inputs, n_hidden, n_outputs, X_train, y_train,
                                n_samples, n_epochs, learning_rate, verbose, seed,
                                timeout \\ 60_000) do
    with :ok <- validate_positive("n_inputs", n_inputs),
         :ok <- validate_positive("n_hidden", n_hidden),
         :ok <- validate_positive("n_outputs", n_outputs),
         :ok <- validate_positive("n_samples", n_samples),
         :ok <- validate_positive("n_epochs", n_epochs),
         :ok <- validate_positive("learning_rate", learning_rate),
         :ok <- validate_binary_size(X_train, n_samples * n_inputs * 8, "X_train"),
         :ok <- validate_binary_size(y_train, n_samples * n_outputs * 8, "y_train") do

      execute_with_timeout(fn ->
        result = ASM.fp_neural_network_train(n_inputs, n_hidden, n_outputs,
                                               X_train, y_train, n_samples,
                                               n_epochs, learning_rate, verbose, seed)
        {:ok, result}
      end, timeout)
    end
  end

  # ==================== Statistics ====================

  @doc """
  Safe correlation with validation.
  """
  def fp_correlation_f64(x, y, n) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_binary_size(x, n * 8, "x"),
         :ok <- validate_binary_size(y, n * 8, "y") do
      {:ok, ASM.fp_correlation_f64(x, y, n)}
    end
  end

  @doc """
  Safe covariance with validation.
  """
  def fp_covariance_f64(x, y, n) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_binary_size(x, n * 8, "x"),
         :ok <- validate_binary_size(y, n * 8, "y") do
      {:ok, ASM.fp_covariance_f64(x, y, n)}
    end
  end

  # ==================== Rolling Window Operations ====================

  @doc """
  Safe rolling mean with window validation.
  """
  def fp_rolling_mean_f64(data, n, window, size_output) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_positive("window", window),
         :ok <- validate_window_size(window, n),
         :ok <- validate_binary_size(data, n * 8, "data") do
      {:ok, ASM.fp_rolling_mean_f64(data, n, window, size_output)}
    end
  end

  @doc """
  Safe rolling sum with window validation.
  """
  def fp_rolling_sum_f64(data, n, window, size_output) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_positive("window", window),
         :ok <- validate_window_size(window, n),
         :ok <- validate_binary_size(data, n * 8, "data") do
      {:ok, ASM.fp_rolling_sum_f64(data, n, window, size_output)}
    end
  end

  # ==================== Vector Operations ====================

  @doc """
  Safe dot product with validation.
  """
  def fp_fold_dotp_f64(a, b, n) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_binary_size(a, n * 8, "a"),
         :ok <- validate_binary_size(b, n * 8, "b") do
      {:ok, ASM.fp_fold_dotp_f64(a, b, n)}
    end
  end

  @doc """
  Safe AXPY (a * x + y) with validation.
  """
  def fp_map_axpy_f64(x, y, size_out, n, c) do
    with :ok <- validate_positive("n", n),
         :ok <- validate_binary_size(x, n * 8, "x"),
         :ok <- validate_binary_size(y, n * 8, "y") do
      {:ok, ASM.fp_map_axpy_f64(x, y, size_out, n, c)}
    end
  end

  # ==================== Validation Helpers ====================

  defp validate_positive(name, value) when is_integer(value) and value > 0, do: :ok
  defp validate_positive(name, value) when is_float(value) and value > 0.0, do: :ok
  defp validate_positive(name, _value) do
    {:error, "#{name} must be positive"}
  end

  defp validate_tolerance(tol) when is_float(tol) and tol > 0.0 and tol < 1.0, do: :ok
  defp validate_tolerance(_tol) do
    {:error, "tolerance must be between 0 and 1"}
  end

  defp validate_k_vs_n(k, n) when k <= n, do: :ok
  defp validate_k_vs_n(k, n) do
    {:error, "k (#{k}) must be <= n (#{n})"}
  end

  defp validate_range(value, min, max, _msg) when value >= min and value <= max, do: :ok
  defp validate_range(_value, _min, _max, msg) do
    {:error, msg}
  end

  defp validate_window_size(window, n) when window <= n, do: :ok
  defp validate_window_size(window, n) do
    {:error, "window size (#{window}) must be <= data length (#{n})"}
  end

  defp validate_binary_size(binary, expected_size, name) when is_binary(binary) do
    actual_size = byte_size(binary)
    if actual_size == expected_size do
      :ok
    else
      {:error, "#{name}: expected #{expected_size} bytes, got #{actual_size} bytes"}
    end
  end
  defp validate_binary_size(_not_binary, _expected_size, name) do
    {:error, "#{name} must be a binary"}
  end

  # ==================== Timeout Wrapper ====================

  defp execute_with_timeout(fun, timeout) do
    task = Task.async(fun)

    case Task.yield(task, timeout) || Task.shutdown(task) do
      {:ok, {:ok, result}} -> {:ok, result}
      {:ok, {:error, reason}} -> {:error, reason}
      nil -> {:error, :timeout}
      {:exit, reason} -> {:error, {:crashed, reason}}
    end
  rescue
    e -> {:error, {:exception, Exception.message(e)}}
  end

  # ==================== Batch Operations ====================

  @doc """
  Execute multiple operations in parallel with individual timeouts.
  Returns list of results in same order as inputs.
  """
  def parallel_map(items, fun, timeout \\ @default_timeout) do
    tasks = Enum.map(items, fn item ->
      Task.async(fn ->
        try do
          {:ok, fun.(item)}
        rescue
          e -> {:error, {:exception, Exception.message(e)}}
        end
      end)
    end)

    Enum.map(tasks, fn task ->
      case Task.yield(task, timeout) || Task.shutdown(task) do
        {:ok, result} -> result
        nil -> {:error, :timeout}
      end
    end)
  end
end
