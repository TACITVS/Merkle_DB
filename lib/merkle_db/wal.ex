defmodule MerkleDb.WAL do
  @moduledoc """
  Write-Ahead Log for MerkleDB.

  Provides crash-safe durability by logging all operations before they are applied.
  Uses Erlang Term Format for flexible data encoding and CRC32 for integrity.
  """

  use GenServer
  require Logger

  @wal_magic "MWAL"
  @wal_version 3
  @entry_upsert 0x01
  @entry_delete 0x02
  @entry_commit 0x03
  @header_size 9  # CRC(4) + Type(1) + Len(4)

  defmodule State do
    @moduledoc false
    defstruct [:path, :fd, :sync_mode, :bytes_written, :entry_count]
  end

  # Client API

  def start_link(opts \\ []) do
    path = Keyword.get(opts, :path) || Application.get_env(:merkle_db, :wal_path, "data/wal.bin")
    GenServer.start_link(__MODULE__, {path, opts}, name: __MODULE__)
  end

  @doc "Open a WAL file directly (for StorageEngine or Replay)"
  def open(path, opts \\ []) do
    GenServer.start_link(__MODULE__, {path, opts})
  end

  def append_upsert(data), do: append_upsert(__MODULE__, data)
  def append_upsert(wal, data), do: GenServer.call(wal, {:append, :upsert, data})

  def append_delete(data), do: append_delete(__MODULE__, data)
  def append_delete(wal, data), do: GenServer.call(wal, {:append, :delete, data})

  def append_commit(data), do: append_commit(__MODULE__, data)
  def append_commit(wal, data), do: GenServer.call(wal, {:append, :commit, data})

  def sync(wal \\ __MODULE__) do
    GenServer.call(wal, :sync)
  end

  def close(wal \\ __MODULE__) do
    GenServer.call(wal, :close)
  end

  def replay(path) when is_binary(path) do
    if File.exists?(path) do
      case File.open(path, [:read, :binary]) do
        {:ok, fd} ->
          result = do_replay(fd, [])
          File.close(fd)
          result
        error -> error
      end
    else
      {:ok, []}
    end
  end

  def stats(wal \\ __MODULE__) do
    GenServer.call(wal, :stats)
  end

  def reset(wal \\ __MODULE__) do
    GenServer.call(wal, :reset)
  end

  # Server Callbacks

  @impl true
  def init({path, opts}) do
    sync_mode = Keyword.get(opts, :sync_mode, :sync)
    case open_or_create(path) do
      {:ok, fd, bytes} ->
        {:ok, %State{path: path, fd: fd, sync_mode: sync_mode, bytes_written: bytes, entry_count: 0}}
      {:error, reason} -> {:stop, reason}
    end
  end

  @impl true
  def handle_call({:append, type, data}, _from, state) do
    case write_entry(state.fd, type, data) do
      {:ok, bytes} ->
        if state.sync_mode == :sync, do: :file.sync(state.fd)
        {:reply, :ok, %{state | bytes_written: state.bytes_written + bytes, entry_count: state.entry_count + 1}}
      error -> {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:sync, _from, state) do
    {:reply, :file.sync(state.fd), state}
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

  @impl true
  def handle_call(:reset, _from, state) do
    header = <<@wal_magic, @wal_version::8, 0::24>>
    :ok = :file.pwrite(state.fd, 0, header)
    :ok = :file.truncate(state.fd)
    {:ok, 8} = :file.position(state.fd, 8)
    {:reply, :ok, %{state | bytes_written: 8, entry_count: 0}}
  end

  # Internal

  defp open_or_create(path) do
    File.mkdir_p!(Path.dirname(path))
    case File.open(path, [:read, :write, :binary]) do
      {:ok, fd} ->
        case verify_or_init_header(fd) do
          {:ok, bytes} -> {:ok, _} = :file.position(fd, :eof); {:ok, fd, bytes}
          error -> error
        end
      error -> error
    end
  end

  defp verify_or_init_header(fd) do
    case :file.pread(fd, 0, 8) do
      {:ok, <<@wal_magic, @wal_version::8, _::24>>} -> {:ok, size} = :file.position(fd, :eof); {:ok, size}
      _ ->
        header = <<@wal_magic, @wal_version::8, 0::24>>
        :ok = :file.pwrite(fd, 0, header)
        :file.truncate(fd)
        {:ok, 8}
    end
  end

  defp write_entry(fd, type, data) do
    type_byte = case type do
      :upsert -> @entry_upsert
      :delete -> @entry_delete
      :commit -> @entry_commit
    end
    payload = :erlang.term_to_binary(data)
    len = byte_size(payload)
    crc = :erlang.crc32(<<type_byte::8, len::little-32, payload::binary>>)
    entry = <<crc::little-32, type_byte::8, len::little-32, payload::binary>>
    case :file.write(fd, entry) do
      :ok -> {:ok, byte_size(entry)}
      error -> error
    end
  end

  defp do_replay(fd, acc) do
    if acc == [], do: {:ok, _} = :file.position(fd, 8)
    case read_entry(fd) do
      {:ok, entry} -> do_replay(fd, [entry | acc])
      :eof -> {:ok, Enum.reverse(acc)}
      {:error, :corrupted} -> Logger.warning("WAL corruption, stopping replay"); {:ok, Enum.reverse(acc)}
      error -> error
    end
  end

  defp read_entry(fd) do
    case :file.read(fd, @header_size) do
      {:ok, <<crc::little-32, type_byte::8, len::little-32>>} ->
        case :file.read(fd, len) do
          {:ok, payload} when byte_size(payload) == len ->
            if crc == :erlang.crc32(<<type_byte::8, len::little-32, payload::binary>>) do
              type = case type_byte do
                @entry_upsert -> :upsert
                @entry_delete -> :delete
                @entry_commit -> :commit
              end
              {:ok, {type, :erlang.binary_to_term(payload)}}
            else
              {:error, :corrupted}
            end
          _ -> {:error, :corrupted}
        end
      {:ok, <<>>} -> :eof
      :eof -> :eof
      _ -> {:error, :corrupted}
    end
  end
end
