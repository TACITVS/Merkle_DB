defmodule MerkleDb.FP.Job do
  @moduledoc """
  Async job wrapper for FP/ASM operations.
  Wraps native calls in a Task-based job abstraction for IndexBuilder.
  """

  defstruct [:task, :status, :result, :started_at]

  alias MerkleDb.ASM

  @type t :: %__MODULE__{
    task: Task.t() | nil,
    status: :queued | :running | :done | :error | :cancelled,
    result: term(),
    started_at: integer()
  }

  @doc """
  Start an async kmeans job.
  Returns a job struct that can be polled for status/result.
  """
  def start_kmeans(data_bin, count, dim, k, max_iter, tol, seed \\ nil) when is_binary(data_bin) do
    seed = seed || :rand.uniform(1_000_000)

    task = Task.async(fn ->
      try do
        result = ASM.fp_kmeans_f64(data_bin, count, dim, k, max_iter, tol, seed)
        {:ok, result}
      rescue
        e -> {:error, Exception.message(e)}
      end
    end)

    %__MODULE__{
      task: task,
      status: :running,
      result: nil,
      started_at: System.monotonic_time(:millisecond)
    }
  end

  @doc """
  Get current status of a job.
  Returns a map with :status, :progress (0-100), :elapsed_ms
  """
  def status(%__MODULE__{} = job) do
    elapsed = System.monotonic_time(:millisecond) - job.started_at

    cond do
      job.status == :cancelled ->
        %{status: :cancelled, progress: 0, elapsed_ms: elapsed}

      job.status == :done ->
        %{status: :done, progress: 100, elapsed_ms: elapsed}

      job.status == :error ->
        %{status: :error, progress: 0, elapsed_ms: elapsed}

      job.task && Task.yield(job.task, 0) != nil ->
        %{status: :done, progress: 100, elapsed_ms: elapsed}

      true ->
        %{status: :running, progress: 50, elapsed_ms: elapsed}  # Approximate
    end
  end

  @doc """
  Get the result of a completed job.
  Returns {:ok, result} | {:error, reason} | {:pending, job}
  """
  def result(%__MODULE__{status: :cancelled}), do: {:error, :cancelled}
  def result(%__MODULE__{status: :error, result: reason}), do: {:error, reason}
  def result(%__MODULE__{result: result}) when result != nil, do: result
  def result(%__MODULE__{task: task} = job) do
    case Task.yield(task, 0) do
      {:ok, result} -> result
      {:exit, reason} -> {:error, reason}
      nil -> {:pending, job}
    end
  end

  @doc """
  Cancel a running job.
  """
  def cancel(%__MODULE__{task: task} = job) when task != nil do
    Task.shutdown(task, :brutal_kill)
    %{job | status: :cancelled, task: nil}
  end
  def cancel(job), do: job

  @doc """
  Await job completion with timeout.
  """
  def await(%__MODULE__{task: task} = job, timeout \\ 60_000) do
    case Task.yield(task, timeout) || Task.shutdown(task) do
      {:ok, result} ->
        %{job | status: :done, result: result, task: nil}
      {:exit, reason} ->
        %{job | status: :error, result: {:error, reason}, task: nil}
      nil ->
        %{job | status: :error, result: {:error, :timeout}, task: nil}
    end
  end
end
