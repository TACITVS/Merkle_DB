defmodule MerkleDb.LoadControlTest do
  use ExUnit.Case, async: false

  defp ensure_started(child_spec) do
    case start_supervised(child_spec) do
      {:ok, pid} -> pid
      {:error, {:already_started, pid}} -> pid
      {:error, {{:already_started, pid}, _child}} -> pid
      {:error, {:already_present, pid}} -> pid
      {:error, {{:already_present, pid}, _child}} -> pid
    end
  end

  defp wait_until(fun, timeout_ms \\ 500) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_wait(fun, deadline)
  end

  defp do_wait(fun, deadline) do
    if fun.() do
      true
    else
      if System.monotonic_time(:millisecond) < deadline do
        Process.sleep(10)
        do_wait(fun, deadline)
      else
        false
      end
    end
  end

  defp ensure_telemetry_started do
    case Application.ensure_all_started(:telemetry) do
      {:ok, _apps} -> :ok
      {:error, {:already_started, :telemetry}} -> :ok
      {:error, {_app, _reason}} -> :ok
    end
  end

  setup do
    ensure_started(MerkleDb.KV)
    ensure_started(MerkleDb.LoadGenerator)
    _ = MerkleDb.LoadGenerator.stop_load()
    :ok
  end

  test "stop_if_active stops the load generator" do
    {:ok, _} = MerkleDb.LoadGenerator.start_load(1)
    assert MerkleDb.LoadGenerator.active?()

    _ = MerkleDb.LoadGenerator.stop_if_active()
    assert wait_until(fn -> MerkleDb.LoadGenerator.active?() == false end)
  end

  test "telemetry metrics include load status" do
    ensure_telemetry_started()
    ensure_started(MerkleDb.TelemetryAggregator)
    {:ok, _} = MerkleDb.LoadGenerator.start_load(1)
    assert wait_until(fn -> MerkleDb.LoadGenerator.active?() end)

    metrics = MerkleDb.TelemetryAggregator.get_metrics()
    assert is_map(metrics.load_status)
    assert metrics.load_status.active == true
    assert metrics.stale == false
    assert is_map(metrics.build_info)
    assert is_binary(metrics.build_info.commit)
  end

  test "stop increments run id and zeroes qps" do
    {:ok, _} = MerkleDb.LoadGenerator.start_load(5)
    status = MerkleDb.LoadGenerator.status_snapshot()
    assert status.active == true

    _ = MerkleDb.LoadGenerator.stop_load()
    updated = MerkleDb.LoadGenerator.status_snapshot()
    assert updated.active == false
    assert updated.actual_qps == 0.0
    assert updated.run_id == status.run_id + 1
  end

  test "telemetry qps resets between polls" do
    ensure_telemetry_started()
    ensure_started(MerkleDb.TelemetryAggregator)

    GenServer.cast(MerkleDb.TelemetryAggregator, {:record_query, 10.0})
    assert wait_until(fn ->
      MerkleDb.TelemetryAggregator.get_metrics().query_metrics.total_queries >= 1
    end)

    metrics1 = MerkleDb.TelemetryAggregator.get_metrics()
    metrics2 = MerkleDb.TelemetryAggregator.get_metrics()
    assert metrics1.query_metrics.qps_current >= 0.0
    assert metrics2.query_metrics.qps_current == 0.0
  end

  test "telemetry returns stale snapshot when aggregator is down" do
    ensure_telemetry_started()

    metrics = MerkleDb.TelemetryAggregator.get_metrics()
    assert metrics.stale == true
    assert is_map(metrics.load_status)
  end

  test "bootstrap start stops the load generator" do
    ensure_started({Task.Supervisor, name: MerkleDb.TaskSupervisor})
    ensure_started(MerkleDb.TextStore)
    ensure_started(MerkleDb.Bootstrap)
    {:ok, _} = MerkleDb.LoadGenerator.start_load(1)
    assert MerkleDb.LoadGenerator.active?()

    {:ok, _} =
      MerkleDb.Bootstrap.start(
        mode: :auto,
        build_index: false,
        save_snapshot: false,
        poll_interval_ms: 10
      )

    assert wait_until(fn -> MerkleDb.LoadGenerator.active?() == false end)
  end
end
