defmodule MerkleDb.FPDispatcherTest do
  use ExUnit.Case, async: false

  defmodule Helpers do
    def add(a, b), do: a + b
    def fail, do: raise "boom"
    def sleep(ms) do
      Process.sleep(ms)
      :ok
    end
  end

  defp ensure_started(child_spec) do
    case start_supervised(child_spec) do
      {:ok, pid} -> pid
      {:error, {:already_started, pid}} -> pid
      {:error, {{:already_started, pid}, _child}} -> pid
      {:error, {:already_present, pid}} -> pid
      {:error, {{:already_present, pid}, _child}} -> pid
    end
  end

  setup do
    ensure_started({Task.Supervisor, name: MerkleDb.TaskSupervisor})
    ensure_started(MerkleDb.FPDispatcher)
    :ok
  end

  test "call executes in sync mode with override" do
    result = MerkleDb.FPDispatcher.call(:add, [1, 2], module: Helpers, mode: "sync")
    assert result == 3
  end

  test "async returns a job id that can be awaited" do
    job_id = MerkleDb.FPDispatcher.async(:add, [2, 3], module: Helpers, mode: "async")
    assert is_integer(job_id)
    assert MerkleDb.FPDispatcher.await(job_id, 1_000) == {:ok, 5}
  end

  test "dispatch_many executes jobs with concurrency" do
    jobs = [
      {:add, [1, 2]},
      %{name: :add, args: [3, 4]}
    ]

    results = MerkleDb.FPDispatcher.dispatch_many(jobs, module: Helpers, mode: "async")
    assert results == [{:ok, 3}, {:ok, 7}]
  end

  test "result returns completed value after await" do
    job_id = MerkleDb.FPDispatcher.async(:add, [5, 6], module: Helpers, mode: "async")
    assert MerkleDb.FPDispatcher.await(job_id, 1_000) == {:ok, 11}
    assert MerkleDb.FPDispatcher.result(job_id) == {:ok, 11}
  end

  test "cancel stops queued jobs" do
    max = System.schedulers_online()
    :ok = MerkleDb.FPDispatcher.configure(max_concurrency: 1)

    on_exit(fn ->
      _ = MerkleDb.FPDispatcher.configure(max_concurrency: max)
    end)

    job1 = MerkleDb.FPDispatcher.async(:sleep, [200], module: Helpers, mode: "async")
    job2 = MerkleDb.FPDispatcher.async(:add, [1, 2], module: Helpers, mode: "async")

    assert {:ok, :queued} = MerkleDb.FPDispatcher.cancel(job2)
    assert MerkleDb.FPDispatcher.await(job2, 1_000) == {:error, {:canceled, :queued}}
    assert MerkleDb.FPDispatcher.await(job1, 1_000) == {:ok, :ok}
  end

  test "status tracks failures" do
    assert_raise RuntimeError, "boom", fn ->
      MerkleDb.FPDispatcher.call(:fail, [], module: Helpers, mode: "sync")
    end

    status = MerkleDb.FPDispatcher.status()
    assert status.total_failed >= 1
    assert status.last_error != nil
  end
end
