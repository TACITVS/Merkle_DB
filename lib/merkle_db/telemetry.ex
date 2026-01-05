defmodule MerkleDb.Telemetry do
  @moduledoc """
  Telemetry instrumentation for MerkleDb operations.
  Tracks performance metrics, errors, and usage patterns.
  """

  @doc """
  Attach telemetry handlers for MerkleDb events.
  """
  def attach_handlers do
    events = [
      [:merkle_db, :query, :execute],
      [:merkle_db, :tree, :insert],
      [:merkle_db, :tree, :insert_batch],
      [:merkle_db, :analytics, :build_ivf],
      [:merkle_db, :analytics, :pca],
      [:merkle_db, :asm, :nif_call],
      [:merkle_db, :cache, :hit],
      [:merkle_db, :cache, :miss]
    ]

    :telemetry.attach_many(
      "merkle-db-handler",
      events,
      &__MODULE__.handle_event/4,
      nil
    )
  end

  @doc """
  Handle telemetry events and log them.
  """
  def handle_event([:merkle_db, :query, :execute], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000_000
    mode = Map.get(metadata, :type) || Map.get(metadata, :mode)

    IO.puts("""
    [MerkleDB Query]
      Duration: #{Float.round(duration_ms, 2)}ms
      Results: #{metadata[:result_count]}
      Mode: #{inspect(mode)}
      K: #{inspect(metadata[:k])}
      Threshold: #{inspect(metadata[:threshold])}
    """)
  end

  def handle_event([:merkle_db, :tree, :insert], measurements, metadata, _config) do
    duration_us = measurements.duration / 1_000

    if duration_us > 1000 do  # Log if > 1ms
      IO.puts("[MerkleDB Insert] Duration: #{Float.round(duration_us / 1000, 2)}ms, Count: #{metadata[:count]}")
    end
  end

  def handle_event([:merkle_db, :tree, :insert_batch], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000_000
    throughput = metadata[:batch_size] / (duration_ms / 1000)

    IO.puts("""
    [MerkleDB Batch Insert]
      Duration: #{Float.round(duration_ms, 2)}ms
      Batch Size: #{metadata[:batch_size]}
      Throughput: #{Float.round(throughput, 0)} vectors/sec
    """)
  end

  def handle_event([:merkle_db, :analytics, :build_ivf], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000_000

    IO.puts("""
    [MerkleDB IVF Index]
      Duration: #{Float.round(duration_ms, 2)}ms
      Vectors: #{metadata[:count]}
      Clusters: #{metadata[:k]}
      Converged: #{metadata[:converged]}
    """)
  end

  def handle_event([:merkle_db, :analytics, :pca], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000_000

    IO.puts("""
    [MerkleDB PCA]
      Duration: #{Float.round(duration_ms, 2)}ms
      Vectors: #{metadata[:count]}
      Components: #{metadata[:n_components]}
      Converged: #{metadata[:converged]}
    """)
  end

  def handle_event([:merkle_db, :asm, :nif_call], measurements, metadata, _config) do
    duration_us = measurements.duration / 1_000

    # Only log slow NIF calls (>10ms)
    if duration_us > 10_000 do
      IO.puts("[MerkleDB NIF] Slow call: #{metadata[:function]} took #{Float.round(duration_us / 1000, 2)}ms")
    end
  end

  def handle_event([:merkle_db, :cache, :hit], _measurements, _metadata, _config) do
    # Silent, just count
    :ok
  end

  def handle_event([:merkle_db, :cache, :miss], _measurements, _metadata, _config) do
    # Silent, just count
    :ok
  end

  # ==================== Instrumentation Helpers ====================

  @doc """
  Wrap a function with telemetry span.

  ## Example
      Telemetry.span([:merkle_db, :query, :execute], %{k: 10}, fn ->
        result = execute_query(...)
        {result, %{result_count: length(result)}}
      end)
  """
  def span(event_name, metadata, fun) do
    start_time = System.monotonic_time()

    try do
      {result, extra_metadata} = fun.()

      duration = System.monotonic_time() - start_time
      measurements = %{duration: duration}
      full_metadata = Map.merge(metadata, extra_metadata)

      :telemetry.execute(event_name, measurements, full_metadata)

      result
    rescue
      e ->
        duration = System.monotonic_time() - start_time
        measurements = %{duration: duration}
        error_metadata = Map.merge(metadata, %{error: Exception.message(e)})

        :telemetry.execute(event_name ++ [:error], measurements, error_metadata)

        reraise e, __STACKTRACE__
    end
  end

  @doc """
  Simple event emission without timing.
  """
  def emit(event_name, measurements \\ %{}, metadata \\ %{}) do
    :telemetry.execute(event_name, measurements, metadata)
  end
end
