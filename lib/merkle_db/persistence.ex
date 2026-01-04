defmodule MerkleDb.Persistence do
  @moduledoc """
  Snapshot persistence for MerkleDb.Tree with integrity checks and atomic writes.
  Uses BLAKE3 for checksums (3-5x faster than SHA-256).
  """

  alias MerkleDb.ASM

  @magic <<77, 68, 66, 83>>
  @version 1
  @current_name "snapshot-current.bin"
  @backup_name "snapshot-prev.bin"

  def snapshot_dir do
    System.get_env("MERKLE_DB_SNAPSHOT_DIR") ||
      Application.get_env(:merkle_db, :snapshot_dir) ||
      Path.join(File.cwd!(), "data")
  end

  def snapshot_path, do: Path.join(snapshot_dir(), @current_name)
  def backup_path, do: Path.join(snapshot_dir(), @backup_name)

  def snapshot_info do
    path = snapshot_path()

    case File.stat(path) do
      {:ok, stat} ->
        %{
          exists: true,
          path: path,
          size_bytes: stat.size,
          modified_at_ms: datetime_to_unix_ms(stat.mtime)
        }

      {:error, _} ->
        %{exists: false, path: path}
    end
  end

  def exists?, do: File.exists?(snapshot_path())

  def save(tree, opts \\ []) do
    compress = Keyword.get(opts, :compress, true)
    label = Keyword.get(opts, :label, "snapshot")
    meta = build_meta(tree, label)
    payload = encode_payload(meta, tree, compress)
    checksum = ASM.fp_blake3_hash(payload)
    data = <<@magic::binary, @version::unsigned-32, checksum::binary-32, payload::binary>>

    case :global.trans({__MODULE__, :snapshot}, fn -> write_atomic(data) end) do
      :ok -> {:ok, meta}
      {:error, reason} -> {:error, reason}
      other -> {:error, other}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  def save_async(tree, opts \\ []) do
    Task.Supervisor.start_child(MerkleDb.TaskSupervisor, fn -> save(tree, opts) end)
  end

  def load(opts \\ []) do
    path = Keyword.get(opts, :path, snapshot_path())

    with {:ok, data} <- File.read(path),
         {:ok, {version, meta, tree}} <- decode_snapshot(data) do
      
      # Upgrade tree structure if needed
      tree = MerkleDb.Tree.rebuild_aux_data(tree)
      
      with :ok <- validate_tree(tree) do
        {:ok, %{tree: tree, meta: meta, version: version, path: path}}
      end
    else
      {:error, reason} -> {:error, reason}
    end
  end

  def delete do
    _ = delete_file(snapshot_path())
    _ = delete_file(backup_path())
    :ok
  end

  defp encode_payload(meta, tree, compress) do
    options = if compress, do: [:compressed], else: []
    :erlang.term_to_binary({@version, meta, tree}, options)
  end

  defp decode_snapshot(<<magic::binary-4, version::unsigned-32, checksum::binary-32, payload::binary>>) do
    with true <- magic == @magic || {:error, :invalid_snapshot},
         true <- version == @version || {:error, :version_mismatch},
         true <- ASM.fp_blake3_hash(payload) == checksum || {:error, :checksum_mismatch},
         {@version, meta, tree} <- :erlang.binary_to_term(payload, [:safe]) do
      {:ok, {version, meta, tree}}
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, :invalid_payload}
    end
  end

  defp decode_snapshot(_), do: {:error, :invalid_snapshot}

  defp build_meta(tree, label) do
    clusters = if is_map(tree.clusters), do: map_size(tree.clusters), else: 0
    tombstones = if tree.tombstones, do: MapSet.size(tree.tombstones), else: 0

    %{
      version: @version,
      label: label,
      saved_at_ms: System.system_time(:millisecond),
      vectors: tree.count,
      dim: tree.dim,
      indexed: tree.centroids != nil,
      clusters: clusters,
      tombstones: tombstones
    }
  end

  defp write_atomic(data) do
    dir = snapshot_dir()
    File.mkdir_p!(dir)
    tmp_path = Path.join(dir, "#{@current_name}.tmp-#{System.unique_integer([:positive])}")

    with :ok <- File.write(tmp_path, data, [:binary]),
         :ok <- rotate_current(),
         :ok <- File.rename(tmp_path, snapshot_path()) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp rotate_current do
    current = snapshot_path()
    backup = backup_path()

    if File.exists?(current) do
      _ = delete_file(backup)
      File.rename(current, backup)
    else
      :ok
    end
  end

  defp delete_file(path) do
    case File.rm(path) do
      :ok -> :ok
      {:error, :enoent} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_tree(%MerkleDb.Tree{} = tree) do
    cond do
      tree.count < 0 -> {:error, :invalid_count}
      tree.dim < 0 -> {:error, :invalid_dimensions}
      tree.count == 0 -> :ok
      tree.columns == nil -> {:error, :missing_columns}
      not is_tuple(tree.columns) -> {:error, :invalid_columns}
      tuple_size(tree.columns) != tree.dim -> {:error, :column_dimension_mismatch}
      not is_map(tree.keys) -> {:error, :invalid_keys}
      map_size(tree.keys) != tree.count -> {:error, :key_count_mismatch}
      true -> validate_columns(tree)
    end
  end

  defp validate_tree(_), do: {:error, :invalid_tree}

  defp validate_columns(tree) do
    expected_bytes = tree.count * 8

    case Enum.all?(Tuple.to_list(tree.columns), fn col ->
           is_binary(col) and byte_size(col) == expected_bytes
         end) do
      true -> :ok
      false -> {:error, :column_size_mismatch}
    end
  end

  defp datetime_to_unix_ms({date, time}) do
    base = :calendar.datetime_to_gregorian_seconds({{1970, 1, 1}, {0, 0, 0}})
    current = :calendar.datetime_to_gregorian_seconds({date, time})
    (current - base) * 1000
  end
end
