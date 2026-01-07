defmodule MerkleDb.Raft.Machine do
  @moduledoc """
  Raft State Machine for MerkleDb.
  Implements the `ra_machine` behavior to handle replicated commands.
  """
  @behaviour :ra_machine

  alias MerkleDb.Tree

  @impl :ra_machine
  def init(_conf) do
    # Initial state is an empty map of collections
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
        with {:ok, tree} <- get_tree(collections, collection) do
          new_tree = Tree.insert(tree, key, vector, metadata)
          {Map.put(collections, collection, new_tree), :ok}
        else
          err -> {collections, err}
        end

      {:put_batch, collection, pairs} ->
        with {:ok, tree} <- get_tree(collections, collection) do
          new_tree = Tree.insert_batch(tree, pairs)
          {Map.put(collections, collection, new_tree), :ok}
        else
          err -> {collections, err}
        end

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
        if Map.has_key?(collections, collection) do
          {Map.put(collections, collection, Tree.new()), :ok}
        else
          {collections, {:error, :collection_not_found}}
        end

      {:drop_collection, name} ->
        {Map.delete(collections, name), :ok}

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

  # State snapshots for Raft log truncation
  @impl :ra_machine
  def state_enter(_role, _state), do: []

  @impl :ra_machine
  def tick(_time, _state), do: []
end
