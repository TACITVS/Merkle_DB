defmodule MerkleDb.WAL do
  @moduledoc """
  Write-Ahead Log for MerkleDB.

  The WAL provides crash-safe durability by logging all operations
  before they are applied. On recovery, the WAL is replayed to
  restore state.

  ## Format

  Each WAL entry is:
  ```
  +----------+----------+----------+----------+
  | CRC32(4) | Type(1)  | Len(4)   | Data(N)  |
  +----------+----------+----------+----------+
  ```

  Entry types:
  - 0x01: UPSERT - insert/update a record
  - 0x02: DELETE - delete a record
  - 0x03: COMMIT - mark a commit point
  - 0xFF: EOF    - end of valid data

  ## Usage

  ```elixir
  {:ok, wal} = WAL.open("/path/to/wal")
  :ok = WAL.append(wal, :upsert, {id, vector, payload, version})
  :ok = WAL.sync(wal)
  entries = WAL.replay(wal)
  WAL.close(wal)
  ```
  """

  use GenServer

  require Logger

  @wal_magic "MWAL"
  @wal_version 1
  @entry_upsert 0x01
  @entry_delete 0x02
  @entry_commit 0x03
  @entry_eof 0xFF

  @header_size 9  # CRC(4) + Type(1) + Len(4)

  defmodule State do
    @moduledoc false
    defstruct [:path, :fd, :sync_mode, :bytes_written, :entry_count]
  end

  # Client API

  @doc """
  Open or create a WAL file.

  Options:
  - :sync_mode - :sync (fsync after each write) or :batch (fsync on demand)
  """
  def open(path, opts \\ []) do
    GenServer.start_link(__MODULE__, {path, opts})
  end

  @doc """
  Append an upsert operation to the WAL.
  """
  def append_upsert(wal, {id, vector, payload, version}) do
    GenServer.call(wal, {:append, :upsert, {id, vector, payload, version}})
  end

  @doc """
  Append a delete operation to the WAL.
  """
  def append_delete(wal, id) do
    GenServer.call(wal, {:append, :delete, id})
  end

  @doc """
  Append a commit marker to the WAL.
  Returns the byte offset of the commit point.
  """
  def append_commit(wal, snapshot_root) do
    GenServer.call(wal, {:append, :commit, snapshot_root})
  end

  @doc """
  Force sync WAL to disk.
  """
  def sync(wal) do
    GenServer.call(wal, :sync)
  end

  @doc """
  Close the WAL file.
  """
  def close(wal) do
    GenServer.call(wal, :close)
  end

  @doc """
  Replay the WAL from the beginning.
  Returns a list of entries: {:upsert, record} | {:delete, id} | {:commit, root}
  """
  def replay(path) when is_binary(path) do
    case File.open(path, [:read, :binary]) do
      {:ok, fd} ->
        result = do_replay(fd, [])
        File.close(fd)
        result

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  @doc """
  Truncate WAL after a commit point (for compaction).
  """
  def truncate_after_commit(path, commit_offset) do
    case File.open(path, [:read, :write, :binary]) do
      {:ok, fd} ->
        :file.position(fd, commit_offset)
        :file.truncate(fd)
        File.close(fd)
        :ok

      error ->
        error
    end
  end

  @doc """
  Get WAL statistics.
  """
  def stats(wal) do
    GenServer.call(wal, :stats)
  end

  # Server Callbacks

  @impl true
  def init({path, opts}) do
    sync_mode = Keyword.get(opts, :sync_mode, :batch)

    case open_or_create(path) do
      {:ok, fd, bytes_written} ->
        state = %State{
          path: path,
          fd: fd,
          sync_mode: sync_mode,
          bytes_written: bytes_written,
          entry_count: 0
        }

        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:append, type, data}, _from, state) do
    case write_entry(state.fd, type, data) do
      {:ok, bytes} ->
        new_state = %{state |
          bytes_written: state.bytes_written + bytes,
          entry_count: state.entry_count + 1
        }

        # Sync if in sync mode
        if state.sync_mode == :sync do
          :file.sync(state.fd)
        end

        {:reply, :ok, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:sync, _from, state) do
    result = :file.sync(state.fd)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:close, _from, state) do
    :file.sync(state.fd)
    File.close(state.fd)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      path: state.path,
      bytes_written: state.bytes_written,
      entry_count: state.entry_count,
      sync_mode: state.sync_mode
    }
    {:reply, stats, state}
  end

  # Internal Functions

  defp open_or_create(path) do
    # Ensure directory exists
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    case File.open(path, [:read, :write, :binary]) do
      {:ok, fd} ->
        case verify_or_init_header(fd) do
          {:ok, bytes_written} ->
            # Seek to end for appending
            {:ok, _} = :file.position(fd, :eof)
            {:ok, fd, bytes_written}

          {:error, reason} ->
            File.close(fd)
            {:error, reason}
        end

      error ->
        error
    end
  end

  defp verify_or_init_header(fd) do
    case :file.pread(fd, 0, 8) do
      {:ok, <<@wal_magic, version::8, _reserved::24>>} when version == @wal_version ->
        # Valid header, get file size
        {:ok, size} = :file.position(fd, :eof)
        {:ok, size}

      {:ok, <<>>} ->
        # Empty file, write header
        write_new_header(fd)

      :eof ->
        # Empty file (pread returns :eof on empty file)
        write_new_header(fd)

      {:ok, _} ->
        {:error, :invalid_wal_header}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp write_new_header(fd) do
    header = <<@wal_magic, @wal_version::8, 0::24>>
    :ok = :file.pwrite(fd, 0, header)
    {:ok, 8}
  end

  defp write_entry(fd, type, data) do
    type_byte = case type do
      :upsert -> @entry_upsert
      :delete -> @entry_delete
      :commit -> @entry_commit
    end

    encoded_data = encode_entry_data(type, data)
    data_len = byte_size(encoded_data)

    # Calculate CRC over type + len + data
    crc_input = <<type_byte::8, data_len::little-32, encoded_data::binary>>
    crc = :erlang.crc32(crc_input)

    entry = <<crc::little-32, type_byte::8, data_len::little-32, encoded_data::binary>>

    case :file.write(fd, entry) do
      :ok -> {:ok, byte_size(entry)}
      error -> error
    end
  end

  defp encode_entry_data(:upsert, {id, vector, payload, version}) do
    dim = length(vector)
    vector_bytes = MerkleDb.Canonical.encode_vector(vector)
    {:ok, payload_json} = Jason.encode(payload)

    <<
      id::little-unsigned-128,
      dim::little-unsigned-32,
      vector_bytes::binary,
      byte_size(payload_json)::little-unsigned-32,
      payload_json::binary,
      version::little-unsigned-64
    >>
  end

  defp encode_entry_data(:delete, id) do
    <<id::little-unsigned-128>>
  end

  defp encode_entry_data(:commit, snapshot_root) when byte_size(snapshot_root) == 32 do
    snapshot_root
  end

  # Replay logic

  defp do_replay(fd, acc) do
    # Skip header on first call
    if acc == [] do
      {:ok, _} = :file.position(fd, 8)
    end

    case read_entry(fd) do
      {:ok, entry} ->
        do_replay(fd, [entry | acc])

      :eof ->
        {:ok, Enum.reverse(acc)}

      {:error, :corrupted} ->
        # Stop at corruption, return what we have
        # Stop at corruption, return what we have
        if Code.ensure_loaded?(Mix) and Mix.env() == :test do
          Logger.debug("WAL corruption detected, stopping replay")
        else
          Logger.warning("WAL corruption detected, stopping replay")
        end
        {:ok, Enum.reverse(acc)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp read_entry(fd) do
    case :file.read(fd, @header_size) do
      {:ok, <<crc::little-32, type::8, len::little-32>>} ->
        case :file.read(fd, len) do
          {:ok, data} when byte_size(data) == len ->
            # Verify CRC
            expected_crc = :erlang.crc32(<<type::8, len::little-32, data::binary>>)

            if crc == expected_crc do
              decode_entry(type, data)
            else
              {:error, :corrupted}
            end

          {:ok, _} ->
            {:error, :corrupted}

          :eof ->
            {:error, :corrupted}
        end

      {:ok, <<>>} ->
        :eof

      {:ok, _partial} ->
        {:error, :corrupted}

      :eof ->
        :eof

      error ->
        error
    end
  end

  defp decode_entry(@entry_upsert, data) do
    <<
      id::little-unsigned-128,
      dim::little-unsigned-32,
      rest::binary
    >> = data

    vector_size = dim * 4
    <<
      vector_bytes::binary-size(vector_size),
      payload_len::little-unsigned-32,
      payload_json::binary-size(payload_len),
      version::little-unsigned-64
    >> = rest

    vector = MerkleDb.Canonical.decode_vector(vector_bytes, dim)
    {:ok, payload} = Jason.decode(payload_json)

    {:ok, {:upsert, {id, vector, payload, version}}}
  end

  defp decode_entry(@entry_delete, <<id::little-unsigned-128>>) do
    {:ok, {:delete, id}}
  end

  defp decode_entry(@entry_commit, <<snapshot_root::binary-32>>) do
    {:ok, {:commit, snapshot_root}}
  end

  defp decode_entry(@entry_eof, _) do
    :eof
  end

  defp decode_entry(_, _) do
    {:error, :unknown_entry_type}
  end
end
