defmodule MerkleDb.ApiKeyStore do
  @moduledoc """
  ETS-based API key storage and validation.

  Supports multiple API keys with different scopes:
  - :read - Can query data
  - :write - Can insert/update/delete data
  - :admin - Can manage collections, view metrics, access dashboard

  API keys are stored as SHA-256 hashes for security.
  """
  use GenServer
  require Logger

  @table :merkle_db_api_keys
  @scopes [:read, :write, :admin]

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Validate an API key and check if it has the required scope.
  Returns {:ok, key_info} or {:error, reason}.
  """
  @spec validate(String.t(), atom()) :: {:ok, map()} | {:error, atom()}
  def validate(key, required_scope \\ :read) when is_binary(key) and required_scope in @scopes do
    key_hash = hash_key(key)

    case :ets.lookup(@table, key_hash) do
      [{^key_hash, key_info}] ->
        if required_scope in key_info.scopes do
          {:ok, key_info}
        else
          {:error, :insufficient_scope}
        end

      [] ->
        {:error, :invalid_key}
    end
  end

  @doc """
  Check if an API key is valid (any scope).
  """
  @spec valid?(String.t()) :: boolean()
  def valid?(key) when is_binary(key) do
    key_hash = hash_key(key)
    :ets.member(@table, key_hash)
  end

  @doc """
  Add a new API key. Used for dynamic key management.
  """
  @spec add_key(String.t(), String.t(), [atom()]) :: :ok | {:error, atom()}
  def add_key(key, name, scopes \\ [:read, :write]) when is_binary(key) and is_binary(name) do
    GenServer.call(__MODULE__, {:add_key, key, name, scopes})
  end

  @doc """
  Remove an API key by its hash or original value.
  """
  @spec remove_key(String.t()) :: :ok
  def remove_key(key) when is_binary(key) do
    GenServer.call(__MODULE__, {:remove_key, key})
  end

  @doc """
  List all API key names and scopes (not the actual keys).
  """
  @spec list_keys() :: [map()]
  def list_keys do
    :ets.tab2list(@table)
    |> Enum.map(fn {_hash, info} ->
      Map.take(info, [:name, :scopes, :created_at])
    end)
  end

  @doc """
  Get the total number of registered API keys.
  """
  @spec count() :: non_neg_integer()
  def count do
    :ets.info(@table, :size)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Create ETS table
    :ets.new(@table, [:set, :protected, :named_table, read_concurrency: true])

    # Load keys from configuration
    load_keys_from_config()

    Logger.info("ApiKeyStore initialized with #{count()} keys")
    {:ok, %{}}
  end

  @impl true
  def handle_call({:add_key, key, name, scopes}, _from, state) do
    valid_scopes = Enum.filter(scopes, &(&1 in @scopes))

    if valid_scopes == [] do
      {:reply, {:error, :invalid_scopes}, state}
    else
      key_hash = hash_key(key)
      key_info = %{
        name: name,
        scopes: valid_scopes,
        created_at: DateTime.utc_now()
      }

      :ets.insert(@table, {key_hash, key_info})
      Logger.info("API key '#{name}' added with scopes: #{inspect(valid_scopes)}")
      {:reply, :ok, state}
    end
  end

  @impl true
  def handle_call({:remove_key, key}, _from, state) do
    key_hash = hash_key(key)
    :ets.delete(@table, key_hash)
    {:reply, :ok, state}
  end

  # Private Functions

  defp hash_key(key) do
    :crypto.hash(:sha256, key)
  end

  defp load_keys_from_config do
    api_keys = Application.get_env(:merkle_db, :api_keys, [])

    Enum.each(api_keys, fn key_config ->
      key = Map.get(key_config, :key)
      name = Map.get(key_config, :name, "unnamed")
      scopes = Map.get(key_config, :scopes, [:read, :write])

      if key && is_binary(key) && byte_size(key) > 0 do
        key_hash = hash_key(key)
        key_info = %{
          name: name,
          scopes: scopes,
          created_at: DateTime.utc_now()
        }
        :ets.insert(@table, {key_hash, key_info})
      end
    end)
  end
end
