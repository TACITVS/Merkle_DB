defmodule MerkleDb.VectorCache do
  @moduledoc """
  ETS-backed cache for expensive vector operations.
  Reduces redundant computation for repeated queries.
  """

  use GenServer

  @table_name :merkle_vector_cache
  @max_cache_size 10_000  # Maximum cached entries
  @ttl_seconds 3600       # 1 hour TTL

  # ==================== Client API ====================

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Get cached result or compute and cache it.

  ## Example
      VectorCache.get_or_compute({:query, query_vec, threshold}, fn ->
        expensive_similarity_search(query_vec, threshold)
      end)
  """
  def get_or_compute(key, compute_fn) do
    case lookup(key) do
      {:ok, value} ->
        value
      :miss ->
        value = compute_fn.()
        put(key, value)
        value
    end
  end

  @doc """
  Get cached value if present.
  Returns {:ok, value} | :miss
  """
  def lookup(key) do
    case :ets.lookup(@table_name, key) do
      [{^key, value, expires_at}] ->
        if :erlang.monotonic_time(:second) < expires_at do
          {:ok, value}
        else
          # Expired entry
          :ets.delete(@table_name, key)
          :miss
        end
      [] ->
        :miss
    end
  end

  @doc """
  Put value in cache with TTL.
  """
  def put(key, value, ttl_seconds \\ @ttl_seconds) do
    expires_at = :erlang.monotonic_time(:second) + ttl_seconds
    :ets.insert(@table_name, {key, value, expires_at})

    # Evict oldest entries if cache is too large
    maybe_evict()

    :ok
  end

  @doc """
  Clear all cached entries.
  """
  def clear do
    :ets.delete_all_objects(@table_name)
    :ok
  end

  @doc """
  Get cache statistics.
  """
  def stats do
    info = :ets.info(@table_name)
    size = Keyword.get(info, :size, 0)
    memory_words = Keyword.get(info, :memory, 0)
    memory_mb = Float.round(memory_words * :erlang.system_info(:wordsize) / (1024 * 1024), 2)

    %{
      size: size,
      max_size: @max_cache_size,
      memory_mb: memory_mb,
      ttl_seconds: @ttl_seconds
    }
  end

  # ==================== GenServer Callbacks ====================

  @impl true
  def init(_) do
    # Create ETS table: set (unique keys), public (any process can read)
    :ets.new(@table_name, [
      :set,
      :public,
      :named_table,
      read_concurrency: true,
      write_concurrency: true
    ])

    # Schedule periodic cleanup
    schedule_cleanup()

    {:ok, %{}}
  end

  @impl true
  def handle_info(:cleanup, state) do
    cleanup_expired()
    schedule_cleanup()
    {:noreply, state}
  end

  # ==================== Private Helpers ====================

  defp maybe_evict do
    size = :ets.info(@table_name, :size)

    if size > @max_cache_size do
      # Evict oldest 10% of entries
      evict_count = div(@max_cache_size, 10)

      # Get entries sorted by expiration time (oldest first)
      entries =
        :ets.tab2list(@table_name)
        |> Enum.sort_by(fn {_key, _value, expires_at} -> expires_at end)
        |> Enum.take(evict_count)

      # Delete oldest entries
      Enum.each(entries, fn {key, _value, _expires_at} ->
        :ets.delete(@table_name, key)
      end)
    end
  end

  defp cleanup_expired do
    now = :erlang.monotonic_time(:second)

    # Delete all expired entries
    :ets.select_delete(@table_name, [
      {{:"$1", :"$2", :"$3"}, [{:<, :"$3", now}], [true]}
    ])
  end

  defp schedule_cleanup do
    # Run cleanup every 5 minutes
    Process.send_after(self(), :cleanup, 5 * 60 * 1000)
  end
end
