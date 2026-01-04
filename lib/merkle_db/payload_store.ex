defmodule MerkleDb.PayloadStore do
  @moduledoc """
  Payload/metadata storage for vector records.

  Stores JSON-like payload data associated with vector keys,
  enabling filtered queries based on metadata attributes.

  Uses ETS for fast in-memory lookups with optional DETS persistence.
  """
  use GenServer

  @table :payload_store
  @dets_table :payload_store_dets
  @dets_filename "payloads.dets"

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Store a payload for a key.
  """
  @spec put(term(), map()) :: :ok
  def put(key, payload) when is_map(payload) do
    GenServer.cast(__MODULE__, {:put, key, payload})
  end

  @doc """
  Get payload for a key.
  """
  @spec get(term()) :: map() | nil
  def get(key) do
    case :ets.lookup(@table, key) do
      [{^key, payload}] -> payload
      [] -> nil
    end
  end

  @doc """
  Get payloads for multiple keys.
  """
  @spec get_many([term()]) :: %{term() => map()}
  def get_many(keys) do
    keys
    |> Enum.map(fn key ->
      case :ets.lookup(@table, key) do
        [{^key, payload}] -> {key, payload}
        [] -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Map.new()
  end

  @doc """
  Delete payload for a key.
  """
  @spec delete(term()) :: :ok
  def delete(key) do
    GenServer.cast(__MODULE__, {:delete, key})
  end

  @doc """
  Get all payloads.
  """
  @spec get_all() :: [{term(), map()}]
  def get_all do
    :ets.tab2list(@table)
  end

  @doc """
  Count stored payloads.
  """
  @spec count() :: non_neg_integer()
  def count do
    :ets.info(@table, :size)
  end

  @doc """
  Flush to DETS for persistence.
  """
  @spec flush() :: :ok
  def flush do
    GenServer.call(__MODULE__, :flush)
  end

  @doc """
  Clear all payloads.
  """
  @spec clear() :: :ok
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  # --- Server Callbacks ---

  @impl true
  def init(opts) do
    # Create ETS table for fast lookups
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])

    # Optionally load from DETS
    if Keyword.get(opts, :persist, false) do
      load_from_dets()
    end

    {:ok, %{persist: Keyword.get(opts, :persist, false)}}
  end

  @impl true
  def handle_cast({:put, key, payload}, state) do
    :ets.insert(@table, {key, payload})

    if state.persist do
      persist_to_dets(key, payload)
    end

    {:noreply, state}
  end

  @impl true
  def handle_cast({:delete, key}, state) do
    :ets.delete(@table, key)

    if state.persist do
      delete_from_dets(key)
    end

    {:noreply, state}
  end

  @impl true
  def handle_call(:flush, _from, state) do
    if state.persist do
      flush_to_dets()
    end

    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    :ets.delete_all_objects(@table)

    if state.persist do
      clear_dets()
    end

    {:reply, :ok, state}
  end

  # --- DETS Persistence ---

  defp load_from_dets do
    filename = String.to_charlist(@dets_filename)

    case :dets.open_file(@dets_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.to_ets(@dets_table, @table)
        :dets.close(@dets_table)

      {:error, _reason} ->
        :ok
    end
  end

  defp persist_to_dets(key, payload) do
    filename = String.to_charlist(@dets_filename)

    case :dets.open_file(@dets_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.insert(@dets_table, {key, payload})
        :dets.close(@dets_table)

      {:error, _} ->
        :ok
    end
  end

  defp delete_from_dets(key) do
    filename = String.to_charlist(@dets_filename)

    case :dets.open_file(@dets_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.delete(@dets_table, key)
        :dets.close(@dets_table)

      {:error, _} ->
        :ok
    end
  end

  defp flush_to_dets do
    filename = String.to_charlist(@dets_filename)

    case :dets.open_file(@dets_table, file: filename, type: :set) do
      {:ok, _} ->
        :ets.to_dets(@table, @dets_table)
        :dets.close(@dets_table)

      {:error, _} ->
        :ok
    end
  end

  defp clear_dets do
    filename = String.to_charlist(@dets_filename)

    case :dets.open_file(@dets_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.delete_all_objects(@dets_table)
        :dets.close(@dets_table)

      {:error, _} ->
        :ok
    end
  end
end
