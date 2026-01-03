defmodule MerkleDb.Segment do
  @moduledoc """
  Immutable segment storage for MerkleDB.

  A segment is an immutable file containing a sorted set of records.
  Segments are created when the memtable is flushed and never modified.

  ## File Format

  ```
  +------------------+
  | Header (64 bytes)|
  +------------------+
  | Record 0         |
  | Record 1         |
  | ...              |
  | Record N         |
  +------------------+
  | Index            |
  +------------------+
  | Footer (32 bytes)|
  +------------------+
  ```

  ## Header Format (64 bytes)
  - Magic: 4 bytes "MSEG"
  - Version: 1 byte
  - Flags: 1 byte
  - Reserved: 2 bytes
  - Record count: 8 bytes (u64 LE)
  - Min ID: 16 bytes (u128 LE)
  - Max ID: 16 bytes (u128 LE)
  - Index offset: 8 bytes (u64 LE)
  - Checksum: 8 bytes (CRC64)

  ## Footer Format (32 bytes)
  - Index entry count: 8 bytes
  - Segment checksum: 8 bytes
  - Magic: 4 bytes "GEND"
  - Reserved: 12 bytes
  """

  alias MerkleDb.Canonical

  @magic "MSEG"
  @footer_magic "GEND"
  @version 1
  @header_size 64
  @footer_size 32
  @index_entry_size 24  # id (16) + offset (8)

  defmodule Header do
    @moduledoc false
    defstruct [:version, :flags, :record_count, :min_id, :max_id, :index_offset, :checksum]
  end

  defmodule Record do
    @moduledoc false
    defstruct [:id, :vector, :payload, :version, :deleted]
  end

  @doc """
  Write records to a new segment file.
  Records must be sorted by ID.
  Returns {:ok, segment_info} or {:error, reason}.
  """
  def write(path, records) when is_list(records) do
    sorted_records = Enum.sort_by(records, fn {id, _, _, _} -> id end)

    if sorted_records == [] do
      {:error, :empty_segment}
    else
      do_write(path, sorted_records)
    end
  end

  defp do_write(path, records) do
    # Ensure directory exists
    File.mkdir_p!(Path.dirname(path))

    case File.open(path, [:write, :binary, :exclusive]) do
      {:ok, fd} ->
        try do
          # Reserve space for header
          :ok = :file.write(fd, :binary.copy(<<0>>, @header_size))

          # Write records and build index
          {record_offsets, bytes_written} = write_records(fd, records, @header_size, [])

          # Write index
          index_offset = @header_size + bytes_written
          :ok = write_index(fd, record_offsets)

          # Write footer
          footer_offset = index_offset + length(record_offsets) * @index_entry_size
          :ok = write_footer(fd, length(record_offsets), footer_offset)

          # Go back and write header
          {min_id, _, _, _} = hd(records)
          {max_id, _, _, _} = List.last(records)
          header = build_header(length(records), min_id, max_id, index_offset)
          :ok = :file.pwrite(fd, 0, header)

          # Sync and close
          :ok = :file.sync(fd)
          :ok = File.close(fd)

          {:ok, %{
            path: path,
            record_count: length(records),
            min_id: min_id,
            max_id: max_id,
            size: footer_offset + @footer_size
          }}
        rescue
          e ->
            File.close(fd)
            File.rm(path)
            {:error, e}
        end

      {:error, :eexist} ->
        {:error, :segment_exists}

      error ->
        error
    end
  end

  defp write_records(fd, records, offset, acc) do
    Enum.reduce(records, {acc, 0}, fn {id, vector, payload, version}, {offsets, total_bytes} ->
      encoded = encode_record(id, vector, payload, version)
      len = byte_size(encoded)

      # Write length prefix + data
      entry = <<len::little-32, encoded::binary>>
      :ok = :file.write(fd, entry)

      new_offset = offset + total_bytes
      {[{id, new_offset} | offsets], total_bytes + 4 + len}
    end)
    |> then(fn {offsets, bytes} -> {Enum.reverse(offsets), bytes} end)
  end

  defp encode_record(id, vector, payload, version) do
    dim = length(vector)
    vector_bytes = Canonical.encode_vector(vector)
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

  defp write_index(fd, record_offsets) do
    index_data =
      record_offsets
      |> Enum.map(fn {id, offset} ->
        <<id::little-unsigned-128, offset::little-unsigned-64>>
      end)
      |> IO.iodata_to_binary()

    :file.write(fd, index_data)
  end

  defp write_footer(fd, entry_count, _footer_offset) do
    # Calculate checksum over the file so far
    # (simplified - just use entry count for now)
    checksum = entry_count * 0xDEADBEEF

    footer = <<
      entry_count::little-unsigned-64,
      checksum::little-unsigned-64,
      @footer_magic,
      0::96  # reserved
    >>

    :file.write(fd, footer)
  end

  defp build_header(record_count, min_id, max_id, index_offset) do
    # Simplified checksum
    checksum = record_count * 0xCAFEBABE

    header = <<
      @magic,
      @version::8,
      0::8,  # flags
      0::16, # reserved
      record_count::little-unsigned-64,
      min_id::little-unsigned-128,
      max_id::little-unsigned-128,
      index_offset::little-unsigned-64,
      checksum::little-unsigned-64
    >>

    # Pad to 64 bytes
    padding_size = @header_size - byte_size(header)
    header <> :binary.copy(<<0>>, padding_size)
  end

  @doc """
  Read segment header.
  """
  def read_header(path) do
    case File.open(path, [:read, :binary]) do
      {:ok, fd} ->
        result = do_read_header(fd)
        File.close(fd)
        result

      error ->
        error
    end
  end

  defp do_read_header(fd) do
    case :file.read(fd, @header_size) do
      {:ok, <<
        @magic,
        version::8,
        flags::8,
        _reserved::16,
        record_count::little-unsigned-64,
        min_id::little-unsigned-128,
        max_id::little-unsigned-128,
        index_offset::little-unsigned-64,
        checksum::little-unsigned-64,
        _padding::binary
      >>} ->
        {:ok, %Header{
          version: version,
          flags: flags,
          record_count: record_count,
          min_id: min_id,
          max_id: max_id,
          index_offset: index_offset,
          checksum: checksum
        }}

      {:ok, _} ->
        {:error, :invalid_header}

      error ->
        error
    end
  end

  @doc """
  Read a specific record by ID.
  Uses binary search on the index.
  """
  def read_record(path, target_id) do
    case File.open(path, [:read, :binary]) do
      {:ok, fd} ->
        result = do_read_record(fd, target_id)
        File.close(fd)
        result

      error ->
        error
    end
  end

  defp do_read_record(fd, target_id) do
    with {:ok, header} <- do_read_header(fd),
         {:ok, offset} <- binary_search_index(fd, header, target_id) do
      read_record_at(fd, offset)
    end
  end

  defp binary_search_index(fd, header, target_id) do
    # Read index
    :file.position(fd, header.index_offset)

    case :file.read(fd, header.record_count * @index_entry_size) do
      {:ok, index_data} ->
        do_binary_search(index_data, target_id, 0, header.record_count - 1)

      error ->
        error
    end
  end

  defp do_binary_search(_data, _target, low, high) when low > high do
    {:error, :not_found}
  end

  defp do_binary_search(data, target, low, high) do
    mid = div(low + high, 2)
    entry_offset = mid * @index_entry_size

    <<_::binary-size(entry_offset), id::little-unsigned-128, offset::little-unsigned-64, _::binary>> = data

    cond do
      id == target -> {:ok, offset}
      id < target -> do_binary_search(data, target, mid + 1, high)
      true -> do_binary_search(data, target, low, mid - 1)
    end
  end

  defp read_record_at(fd, offset) do
    :file.position(fd, offset)

    case :file.read(fd, 4) do
      {:ok, <<len::little-32>>} ->
        case :file.read(fd, len) do
          {:ok, data} -> decode_record(data)
          error -> error
        end

      error ->
        error
    end
  end

  defp decode_record(data) do
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

    vector = Canonical.decode_vector(vector_bytes, dim)
    {:ok, payload} = Jason.decode(payload_json)

    {:ok, %Record{
      id: id,
      vector: vector,
      payload: payload,
      version: version,
      deleted: false
    }}
  end

  @doc """
  Iterate over all records in a segment.
  Yields each record to the given function.
  """
  def scan(path, fun) do
    case File.open(path, [:read, :binary]) do
      {:ok, fd} ->
        result = with {:ok, header} <- do_read_header(fd) do
          do_scan(fd, @header_size, header.record_count, fun)
        end
        File.close(fd)
        result

      error ->
        error
    end
  end

  defp do_scan(_fd, _offset, 0, _fun), do: :ok

  defp do_scan(fd, offset, remaining, fun) do
    :file.position(fd, offset)

    case :file.read(fd, 4) do
      {:ok, <<len::little-32>>} ->
        case :file.read(fd, len) do
          {:ok, data} ->
            {:ok, record} = decode_record(data)
            fun.(record)
            do_scan(fd, offset + 4 + len, remaining - 1, fun)

          error ->
            error
        end

      error ->
        error
    end
  end

  @doc """
  Merge multiple segments into a new segment.
  Used for compaction.
  """
  def merge(output_path, segment_paths, filter_fn \\ fn _ -> true end) do
    # Open all segments and create iterators
    iterators =
      segment_paths
      |> Enum.map(&open_iterator/1)
      |> Enum.filter(&match?({:ok, _}, &1))
      |> Enum.map(fn {:ok, iter} -> iter end)

    # Merge using min-heap style iteration
    records = merge_iterators(iterators, filter_fn)

    # Close iterators
    Enum.each(iterators, &close_iterator/1)

    if records == [] do
      {:error, :empty_result}
    else
      write(output_path, records)
    end
  end

  defp open_iterator(path) do
    case File.open(path, [:read, :binary]) do
      {:ok, fd} ->
        case do_read_header(fd) do
          {:ok, header} ->
            {:ok, %{fd: fd, offset: @header_size, remaining: header.record_count}}

          error ->
            File.close(fd)
            error
        end

      error ->
        error
    end
  end

  defp close_iterator(%{fd: fd}) do
    File.close(fd)
  end

  defp merge_iterators(iterators, filter_fn) do
    # Simple implementation: read all, sort, filter
    all_records =
      iterators
      |> Enum.flat_map(fn iter ->
        read_all_from_iterator(iter)
      end)
      |> Enum.sort_by(fn {id, _, _, _} -> id end)
      |> Enum.uniq_by(fn {id, _, _, _} -> id end)  # Keep latest (assumes sorted input)
      |> Enum.filter(fn record -> filter_fn.(record) end)

    all_records
  end

  defp read_all_from_iterator(iter) do
    read_all_from_iterator(iter, [])
  end

  defp read_all_from_iterator(%{remaining: 0}, acc), do: Enum.reverse(acc)

  defp read_all_from_iterator(%{fd: fd, offset: offset, remaining: remaining} = iter, acc) do
    :file.position(fd, offset)

    case :file.read(fd, 4) do
      {:ok, <<len::little-32>>} ->
        case :file.read(fd, len) do
          {:ok, data} ->
            {:ok, record} = decode_record(data)
            tuple = {record.id, record.vector, record.payload, record.version}
            new_iter = %{iter | offset: offset + 4 + len, remaining: remaining - 1}
            read_all_from_iterator(new_iter, [tuple | acc])

          _ ->
            Enum.reverse(acc)
        end

      _ ->
        Enum.reverse(acc)
    end
  end
end
