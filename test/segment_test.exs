defmodule MerkleDb.SegmentTest do
  use ExUnit.Case, async: false

  alias MerkleDb.Segment

  @test_dir "test/tmp/segments"

  setup do
    File.rm_rf!(@test_dir)
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      File.rm_rf!(@test_dir)
    end)

    :ok
  end

  defp segment_path(name) do
    Path.join(@test_dir, "#{name}.seg")
  end

  defp make_records(count) do
    Enum.map(1..count, fn i ->
      {i, [i * 1.0, i * 2.0], %{"idx" => i}, 0}
    end)
  end

  describe "write/2" do
    test "creates segment file with records" do
      path = segment_path("basic")
      records = make_records(10)

      {:ok, info} = Segment.write(path, records)

      assert File.exists?(path)
      assert info.record_count == 10
      assert info.min_id == 1
      assert info.max_id == 10
    end

    test "sorts records by ID" do
      path = segment_path("unsorted")
      records = [{5, [5.0], %{}, 0}, {1, [1.0], %{}, 0}, {10, [10.0], %{}, 0}]

      {:ok, info} = Segment.write(path, records)

      assert info.min_id == 1
      assert info.max_id == 10
    end

    test "rejects empty records" do
      path = segment_path("empty")

      assert {:error, :empty_segment} = Segment.write(path, [])
    end

    test "rejects if file exists" do
      path = segment_path("exists")

      {:ok, _} = Segment.write(path, make_records(5))
      assert {:error, :segment_exists} = Segment.write(path, make_records(5))
    end

    test "handles large payloads" do
      path = segment_path("large_payload")
      large_payload = %{"data" => String.duplicate("x", 10_000)}
      records = [{1, [1.0], large_payload, 0}]

      {:ok, info} = Segment.write(path, records)
      assert info.record_count == 1
    end
  end

  describe "read_header/1" do
    test "reads header correctly" do
      path = segment_path("header")
      records = make_records(100)
      {:ok, _} = Segment.write(path, records)

      {:ok, header} = Segment.read_header(path)

      assert header.version == 1
      assert header.record_count == 100
      assert header.min_id == 1
      assert header.max_id == 100
    end

    test "returns error for non-existent file" do
      assert {:error, :enoent} = Segment.read_header(segment_path("missing"))
    end
  end

  describe "read_record/2" do
    test "reads specific record by ID" do
      path = segment_path("read")
      records = make_records(100)
      {:ok, _} = Segment.write(path, records)

      {:ok, record} = Segment.read_record(path, 50)

      assert record.id == 50
      assert record.vector == [50.0, 100.0]
      assert record.payload == %{"idx" => 50}
    end

    test "returns error for missing ID" do
      path = segment_path("missing_id")
      records = make_records(10)
      {:ok, _} = Segment.write(path, records)

      assert {:error, :not_found} = Segment.read_record(path, 999)
    end

    test "reads first record" do
      path = segment_path("first")
      records = make_records(100)
      {:ok, _} = Segment.write(path, records)

      {:ok, record} = Segment.read_record(path, 1)
      assert record.id == 1
    end

    test "reads last record" do
      path = segment_path("last")
      records = make_records(100)
      {:ok, _} = Segment.write(path, records)

      {:ok, record} = Segment.read_record(path, 100)
      assert record.id == 100
    end
  end

  describe "scan/2" do
    test "iterates over all records" do
      path = segment_path("scan")
      records = make_records(10)
      {:ok, _} = Segment.write(path, records)

      results = []
      :ok = Segment.scan(path, fn record ->
        send(self(), {:record, record.id})
      end)

      for i <- 1..10 do
        assert_received {:record, ^i}
      end
    end

    test "handles empty scan function" do
      path = segment_path("scan_noop")
      {:ok, _} = Segment.write(path, make_records(5))

      :ok = Segment.scan(path, fn _ -> :ok end)
    end
  end

  describe "merge/3" do
    test "merges two segments" do
      path1 = segment_path("merge1")
      path2 = segment_path("merge2")
      output = segment_path("merged")

      # First segment: IDs 1-5
      {:ok, _} = Segment.write(path1, make_records(5))

      # Second segment: IDs 6-10
      records2 = Enum.map(6..10, fn i -> {i, [i * 1.0, i * 2.0], %{"idx" => i}, 0} end)
      {:ok, _} = Segment.write(path2, records2)

      {:ok, info} = Segment.merge(output, [path1, path2])

      assert info.record_count == 10
      assert info.min_id == 1
      assert info.max_id == 10
    end

    test "deduplicates overlapping IDs" do
      path1 = segment_path("overlap1")
      path2 = segment_path("overlap2")
      output = segment_path("deduped")

      {:ok, _} = Segment.write(path1, make_records(10))

      # Overlapping segment with IDs 5-15
      records2 = Enum.map(5..15, fn i -> {i, [i * 10.0], %{"new" => true}, 1} end)
      {:ok, _} = Segment.write(path2, records2)

      {:ok, info} = Segment.merge(output, [path1, path2])

      # Should have 15 unique IDs
      assert info.record_count == 15
    end

    test "applies filter function" do
      path1 = segment_path("filter1")
      output = segment_path("filtered")

      {:ok, _} = Segment.write(path1, make_records(10))

      # Filter to keep only even IDs
      filter_fn = fn {id, _, _, _} -> rem(id, 2) == 0 end
      {:ok, info} = Segment.merge(output, [path1], filter_fn)

      assert info.record_count == 5
    end

    test "returns error for empty result" do
      path1 = segment_path("all_filtered")
      output = segment_path("empty_result")

      {:ok, _} = Segment.write(path1, make_records(5))

      # Filter removes everything
      filter_fn = fn _ -> false end
      assert {:error, :empty_result} = Segment.merge(output, [path1], filter_fn)
    end
  end

  describe "stress tests" do
    @tag :slow
    test "handles 10k records" do
      path = segment_path("stress_10k")
      records = make_records(10_000)

      {:ok, info} = Segment.write(path, records)
      assert info.record_count == 10_000

      # Random read
      {:ok, record} = Segment.read_record(path, 5000)
      assert record.id == 5000
    end

    @tag :slow
    test "binary search is efficient" do
      path = segment_path("binary_search")
      records = make_records(100_000)
      {:ok, _} = Segment.write(path, records)

      # Multiple reads should be fast
      for _ <- 1..100 do
        id = :rand.uniform(100_000)
        {:ok, record} = Segment.read_record(path, id)
        assert record.id == id
      end
    end
  end
end
