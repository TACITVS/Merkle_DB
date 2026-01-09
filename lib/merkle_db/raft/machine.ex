defmodule MerkleDb.Raft.Machine do
  @moduledoc """
  Raft State Machine for MerkleDb.
  Implements the `ra_machine` behavior to handle replicated commands.
  """
  @behaviour :ra_machine

  alias MerkleDb.Tree
  require Logger

  @impl :ra_machine
  def init(_conf) do
    # Ensure the directory for this specific server exists if it needs to store segments
    # Ra usually handles this, but the segment_writer was reporting enoent during heavy load.
    # However, Machine init doesn't have easy access to the server ID or data_dir.
    # We'll rely on the default %{} state.
    %{}
  end

  @impl :ra_machine
  def apply(_meta, command, collections) do
    case command do
      {:create_collection, name, opts} ->
        if Map.has_key?(collections, name) do
          {collections, :ok}
        else
          new_tree = Tree.new(opts)
          {Map.put(collections, name, new_tree), :ok}
        end

      {:put, collection, key, vector, metadata} ->
        tree = case get_tree(collections, collection) do
          {:ok, t} -> t
          {:error, :collection_not_found} ->
            # Auto-create collection with inferred settings from vector
            dim = div(byte_size(vector), 8)
            Tree.new(dim: dim, precision: :f64)
        end
        new_tree = Tree.insert(tree, key, vector, metadata)
        {Map.put(collections, collection, new_tree), :ok}

      {:put_batch, collection, pairs} ->
        tree = case get_tree(collections, collection) do
          {:ok, t} -> t
          {:error, :collection_not_found} ->
            # Auto-create collection with inferred settings from first vector
            {_key, first_vec, _meta} = hd(pairs)
            dim = div(byte_size(first_vec), 8)
            Tree.new(dim: dim, precision: :f64)
        end
        new_tree = Tree.insert_batch(tree, pairs)
        {Map.put(collections, collection, new_tree), :ok}

      {:delete, collection, key} ->
        with {:ok, tree} <- get_tree(collections, collection) do
          case Tree.delete(tree, key) do
            {:error, :not_found} -> {collections, {:error, :not_found}}
            new_tree -> {Map.put(collections, collection, new_tree), :ok}
          end
        else
          err -> {collections, err}
        end

      {:set_tree, collection, tree} ->
        {Map.put(collections, collection, tree), :ok}

      {:update_index, collection, new_tree, expected_gen} ->
        with {:ok, current_tree} <- get_tree(collections, collection) do
          if current_tree.generation == expected_gen do
            updated_tree = %{current_tree |
              centroids: new_tree.centroids,
              clusters: new_tree.clusters,
              hnsw: new_tree.hnsw,
              generation: current_tree.generation + 1
            }
            {Map.put(collections, collection, updated_tree), :ok}
          else
            {collections, {:error, :generation_mismatch}}
          end
        else
          err -> {collections, err}
        end

      {:reset, collection} ->
        # Remove collection entirely so it can be auto-created with correct dimensions
        {Map.delete(collections, collection), :ok}

      {:drop_collection, name} ->
        {Map.delete(collections, name), :ok}

      {:get_snapshot, name} ->
        {collections, Map.get(collections, name)}

      {:get_state, _} ->
        {collections, collections}

      _ ->
        {collections, {:error, :unknown_command}}
    end
  end

  # Helpers
  defp get_tree(collections, name) do
    case Map.get(collections, name) do
      nil -> {:error, :collection_not_found}
      tree -> {:ok, tree}
    end
  end

  @impl :ra_machine
  def tick(_time, _state), do: []
end
