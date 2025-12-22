defmodule MerkleDb.FPDispatcherTest do
  use ExUnit.Case, async: false

  defmodule Helpers do
    def add(a, b), do: a + b
    def fail, do: raise "boom"
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

  test "async returns a task that can be awaited" do
    task = MerkleDb.FPDispatcher.async(:add, [2, 3], module: Helpers, mode: "async")
    assert is_struct(task, Task)
    assert Task.await(task) == 5
  end

  test "dispatch_many executes jobs with concurrency" do
    jobs = [
      {:add, [1, 2]},
      %{name: :add, args: [3, 4]}
    ]

    results = MerkleDb.FPDispatcher.dispatch_many(jobs, module: Helpers, mode: "sync", ordered: true)
    assert Enum.map(results, fn {status, value} -> {status, value} end) == [ok: 3, ok: 7]
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
