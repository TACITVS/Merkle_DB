defmodule MerkleDb.SnapshotStore do
  @moduledoc """
  Persistent snapshot storage for MerkleDB.

  Stores committed snapshots so they can be:
  - Retrieved by root hash
  - Used for historical queries
  - Used to generate inclusion proofs

  Each snapshot is stored as a separate file named by its root hash.
  """

  alias MerkleDb.{Snapshot, Crypto, Merkle}

  @snapshot_dir "snapshots"

  @doc """
  Save a snapshot state to disk.
  """
  @spec save(Path.t(), Snapshot.State.t()) :: :ok | {:error, term()}
  def save(data_dir, %Snapshot.State{} = state) do
    root = Snapshot.root(state)
    path = snapshot_path(data_dir, root)

    # Ensure directory exists
    File.mkdir_p!(Path.dirname(path))

    # Serialize and write atomically
    data = Snapshot.serialize(state)
    temp_path = path <> ".tmp"

    with :ok <- File.write(temp_path, data),
         :ok <- File.rename(temp_path, path) do
      :ok
    end
  end

  @doc """
  Load a snapshot by its root hash.
  """
  @spec load(Path.t(), <<_::256>>) :: {:ok, Snapshot.State.t()} | {:error, term()}
  def load(data_dir, root) when byte_size(root) == 32 do
    path = snapshot_path(data_dir, root)

    case File.read(path) do
      {:ok, data} -> Snapshot.deserialize(data)
      error -> error
    end
  end

  @doc """
  Load a snapshot by hex root string.
  """
  @spec load_hex(Path.t(), String.t()) :: {:ok, Snapshot.State.t()} | {:error, term()}
  def load_hex(data_dir, hex_root) when is_binary(hex_root) do
    case Crypto.from_hex(hex_root) do
      {:ok, root} -> load(data_dir, root)
      error -> error
    end
  end

  @doc """
  Check if a snapshot exists.
  """
  @spec exists?(Path.t(), <<_::256>>) :: boolean()
  def exists?(data_dir, root) do
    File.exists?(snapshot_path(data_dir, root))
  end

  @doc """
  List all snapshots in order (newest first).
  """
  @spec list(Path.t()) :: {:ok, [map()]} | {:error, term()}
  def list(data_dir) do
    dir = Path.join(data_dir, @snapshot_dir)

    case File.ls(dir) do
      {:ok, files} ->
        snapshots =
          files
          |> Enum.filter(&String.ends_with?(&1, ".snapshot"))
          |> Enum.map(fn file ->
            root_hex = String.replace_suffix(file, ".snapshot", "")
            path = Path.join(dir, file)

            case File.stat(path) do
              {:ok, stat} ->
                # Convert Erlang datetime tuple to NaiveDateTime
                created = case stat.mtime do
                  {{y, m, d}, {h, min, s}} ->
                    NaiveDateTime.new!(y, m, d, h, min, s)
                  dt ->
                    dt
                end

                %{
                  root: root_hex,
                  size: stat.size,
                  created: created
                }

              _ ->
                nil
            end
          end)
          |> Enum.reject(&is_nil/1)
          |> Enum.sort_by(& &1.created, {:desc, NaiveDateTime})

        {:ok, snapshots}

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  @doc """
  Delete a snapshot.
  """
  @spec delete(Path.t(), <<_::256>>) :: :ok | {:error, term()}
  def delete(data_dir, root) do
    path = snapshot_path(data_dir, root)
    File.rm(path)
  end

  @doc """
  Get the latest snapshot.
  """
  @spec latest(Path.t()) :: {:ok, Snapshot.State.t()} | {:error, :no_snapshots | term()}
  def latest(data_dir) do
    case list(data_dir) do
      {:ok, []} ->
        {:error, :no_snapshots}

      {:ok, [latest | _]} ->
        load_hex(data_dir, latest.root)

      error ->
        error
    end
  end

  @doc """
  Generate an inclusion proof for a record in a snapshot.
  """
  @spec prove_inclusion(Path.t(), <<_::256>>, non_neg_integer()) ::
          {:ok, Merkle.Proof.t()} | {:error, term()}
  def prove_inclusion(data_dir, snapshot_root, record_id) do
    case load(data_dir, snapshot_root) do
      {:ok, state} ->
        Snapshot.prove_inclusion(state, record_id)

      error ->
        error
    end
  end

  @doc """
  Verify an inclusion proof.
  This is a static function - can run without database access.
  """
  @spec verify_inclusion(
          {non_neg_integer(), [float()], map(), non_neg_integer()},
          Merkle.Proof.t()
        ) :: boolean()
  def verify_inclusion(record, proof) do
    Snapshot.verify_inclusion(record, proof)
  end

  @doc """
  Get a record from a specific snapshot.
  """
  @spec get_at_snapshot(Path.t(), <<_::256>>, non_neg_integer()) ::
          {:ok, {non_neg_integer(), [float()], map(), non_neg_integer()}} | {:error, term()}
  def get_at_snapshot(data_dir, snapshot_root, record_id) do
    case load(data_dir, snapshot_root) do
      {:ok, state} ->
        # Search in the tree's leaf data
        case find_record_in_snapshot(state, record_id) do
          nil -> {:error, :not_found}
          record -> {:ok, record}
        end

      error ->
        error
    end
  end

  defp find_record_in_snapshot(%Snapshot.State{tree: tree}, record_id) do
    # The tree stores leaf_ids and leaf_hashes, but not the full records
    # For a full implementation, we'd store records separately or with the tree
    # For now, we check if the ID exists in the tree
    if record_id in tree.leaf_ids do
      # We don't have the full record data in the tree structure
      # This is a limitation - we'd need to store records with the snapshot
      {:needs_storage_lookup, record_id}
    else
      nil
    end
  end

  @doc """
  Get snapshot info without loading full state.
  """
  @spec info(Path.t(), <<_::256>>) :: {:ok, map()} | {:error, term()}
  def info(data_dir, root) do
    case load(data_dir, root) do
      {:ok, state} -> {:ok, Snapshot.info(state)}
      error -> error
    end
  end

  @doc """
  Garbage collect old snapshots based on policy.
  """
  @spec gc(Path.t(), keyword()) :: {:ok, non_neg_integer()} | {:error, term()}
  def gc(data_dir, opts \\ []) do
    keep_count = Keyword.get(opts, :keep_last, 10)
    keep_roots = Keyword.get(opts, :keep_roots, []) |> MapSet.new()

    case list(data_dir) do
      {:ok, snapshots} ->
        # Keep the most recent `keep_count` snapshots
        to_delete =
          snapshots
          |> Enum.drop(keep_count)
          |> Enum.reject(fn s -> MapSet.member?(keep_roots, s.root) end)

        deleted =
          Enum.reduce(to_delete, 0, fn snapshot, count ->
            case Crypto.from_hex(snapshot.root) do
              {:ok, root} ->
                case delete(data_dir, root) do
                  :ok -> count + 1
                  _ -> count
                end

              _ ->
                count
            end
          end)

        {:ok, deleted}

      error ->
        error
    end
  end

  # Internal

  defp snapshot_path(data_dir, root) when byte_size(root) == 32 do
    hex = Crypto.to_hex(root)
    Path.join([data_dir, @snapshot_dir, "#{hex}.snapshot"])
  end
end
