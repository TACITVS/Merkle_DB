defmodule MerkleDb.IndexBuilder do
  use GenServer

  alias MerkleDb.{Analytics, FP, KV, Persistence, Progress, Tree}

  @poll_interval_ms 250
  @min_vectors 10

  # --- Client API ---
  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def start_build(k, opts \\ []) do
    GenServer.call(__MODULE__, {:start, k, opts})
  end

  def status do
    Progress.get_status()
  end

  def cancel do
    GenServer.call(__MODULE__, :cancel)
  end

  # --- Server Callbacks ---

  @impl true
  def init(_) do
    {:ok, %{status: :idle, job: nil}}
  end

  @impl true
  def handle_call({:start, k, opts}, _from, state) do
    if state.status in [:preparing, :queued, :running, :finalizing, :cancelling] do
      {:reply, {:error, :already_running}, state}
    else
      tree = KV.snapshot()
      min_vectors = Keyword.get(opts, :min_vectors, @min_vectors)
      max_iter = Keyword.get(opts, :max_iter, 100)
      tol = Keyword.get(opts, :tol, 1.0e-4)
      seed = Keyword.get(opts, :seed, 42)
      auto_snapshot = Keyword.get(opts, :auto_snapshot, Application.get_env(:merkle_db, :auto_snapshot, true))

      cond do
        tree.count < min_vectors ->
          {:reply, {:error, {:min_vectors, min_vectors}}, state}

        tree.count < k ->
          {:reply, {:error, :k_too_large}, state}

        tree.dim <= 0 ->
          {:reply, {:error, :invalid_dimensions}, state}

        true ->
          started_at_native = System.monotonic_time()
          started_at_ms = System.monotonic_time(:millisecond)

          Progress.report(%{
            status: :preparing,
            phase: :init,
            iter: 0,
            max_iter: max_iter,
            started_at_ms: started_at_ms,
            updated_at_ms: started_at_ms,
            message: "preparing_data",
            op: :fp_kmeans_f64,
            vectors: tree.count,
            dim: tree.dim,
            k: k
          })

          parent = self()

          Task.start(fn ->
            result =
              try do
                {:ok, Tree.flatten(tree)}
              rescue
                e -> {:error, Exception.message(e)}
              end

            send(parent, {:prepared_data, result})
          end)

          new_state = %{
            status: :preparing,
            job: nil,
            tree: tree,
            tree_generation: tree.generation,
            k: k,
            max_iter: max_iter,
            tol: tol,
            seed: seed,
            auto_snapshot: auto_snapshot,
            started_at_native: started_at_native,
            started_at_ms: started_at_ms
          }

          {:reply, {:ok, %{status: :preparing, k: k, vectors: tree.count}}, new_state}
      end
    end
  end

  @impl true
  def handle_call(:cancel, _from, state) do
    case state.status do
      status when status in [:queued, :running] ->
        _ = FP.Job.cancel(state.job)
        Progress.report(%{
          status: :cancelling,
          message: "cancel_requested",
          updated_at_ms: System.monotonic_time(:millisecond)
        })
        {:reply, :ok, state}

      :preparing ->
        Progress.report(%{
          status: :cancelled,
          message: "cancelled",
          updated_at_ms: System.monotonic_time(:millisecond)
        })
        {:reply, :ok, %{status: :idle, job: nil}}

      _ ->
        {:reply, {:error, :not_running}, state}
    end
  end

  @impl true
  def handle_info({:prepared_data, result}, state) do
    if state.status != :preparing do
      {:noreply, state}
    else
      case result do
        {:ok, data_bin} ->
          start_kmeans_job(state, data_bin)

        {:error, reason} ->
          Progress.report(%{
            status: :error,
            phase: :init,
            message: "prepare_failed",
            error: reason,
            updated_at_ms: System.monotonic_time(:millisecond),
            vectors: state.tree.count,
            dim: state.tree.dim,
            k: state.k
          })

          {:noreply, %{status: :idle, job: nil}}
      end
    end
  end

  @impl true
  def handle_info(:poll, state) do
    case Map.get(state, :job) do
      nil ->
        {:noreply, state}

      job ->
        try do
          job_status = FP.Job.status(job)
          update_progress(state, job_status)

          case job_status.status do
            :queued ->
              Process.send_after(self(), :poll, @poll_interval_ms)
              {:noreply, %{state | status: :queued}}

            :running ->
              Process.send_after(self(), :poll, @poll_interval_ms)
              {:noreply, %{state | status: :running}}

            :done ->
              handle_job_done(state, job, job_status)

            :error ->
              reason = fetch_job_error(job)
              Progress.report(%{status: :error, message: "job_error", error: reason})
              {:noreply, %{state | status: :idle, job: nil}}

            :cancelled ->
              Progress.report(%{status: :cancelled, message: "cancelled"})
              {:noreply, %{state | status: :idle, job: nil}}

            _ ->
              Process.send_after(self(), :poll, @poll_interval_ms)
              {:noreply, state}
          end
        rescue
          e ->
            Progress.report(%{status: :error, message: "status_failed", error: Exception.message(e)})
            {:noreply, %{state | status: :idle, job: nil}}
        end
    end
  end

  @impl true
  def handle_info({:finalize_complete, {:ok, {new_tree, metadata}}}, state) do
    elapsed_ms = elapsed_since(state.started_at_ms)

    case KV.update_index(new_tree, state.tree_generation) do
      :ok ->
        if Map.get(state, :auto_snapshot, true) do
          _ = Persistence.save_async(new_tree, label: "ivf_index")
        end

        Progress.report(%{
          status: :done,
          phase: :finalize,
          message: "index_applied",
          elapsed_ms: elapsed_ms,
          updated_at_ms: System.monotonic_time(:millisecond),
          converged: metadata.converged,
          vectors: state.tree.count,
          dim: state.tree.dim,
          k: state.k,
          clusters: map_size(new_tree.clusters)
        })

        {:noreply, %{status: :idle, job: nil}}

      {:error, :stale_tree} ->
        Progress.report(%{
          status: :stale,
          phase: :finalize,
          message: "tree_changed",
          elapsed_ms: elapsed_ms,
          updated_at_ms: System.monotonic_time(:millisecond),
          vectors: state.tree.count,
          dim: state.tree.dim,
          k: state.k
        })

        {:noreply, %{status: :idle, job: nil}}
    end
  end

  @impl true
  def handle_info({:finalize_complete, {:error, reason}}, _state) do
    Progress.report(%{
      status: :error,
      phase: :finalize,
      message: "finalize_failed",
      error: reason,
      updated_at_ms: System.monotonic_time(:millisecond)
    })
    {:noreply, %{status: :idle, job: nil}}
  end

  # --- Helpers ---

  defp handle_job_done(state, job, job_status) do
    case FP.Job.result(job) do
      {:ok, kmeans_res} ->
        Progress.report(%{
          status: :finalizing,
          phase: :finalize,
          iter: job_status.iter,
          max_iter: job_status.max_iter,
          message: "building_clusters",
          updated_at_ms: System.monotonic_time(:millisecond),
          vectors: state.tree.count,
          dim: state.tree.dim,
          k: state.k
        })

        parent = self()

        Task.start(fn ->
          result =
            try do
              {:ok,
               Analytics.build_ivf_index(state.tree, state.k, state.max_iter,
                 kmeans_result: kmeans_res,
                 start_time: state.started_at_native,
                 tol: state.tol,
                 seed: state.seed
               )}
            rescue
              e -> {:error, Exception.message(e)}
            end

          send(parent, {:finalize_complete, result})
        end)

        {:noreply, %{state | status: :finalizing, job: nil}}

      {:error, reason} ->
        Progress.report(%{status: :error, message: "job_result_failed", error: inspect(reason)})
        {:noreply, %{state | status: :idle, job: nil}}
    end
  end

  defp update_progress(state, job_status) do
    Progress.report(%{
      status: job_status.status,
      phase: job_status.phase,
      iter: job_status.iter,
      max_iter: job_status.max_iter,
      started_at_ms: job_status.started_at_ms || state.started_at_ms,
      updated_at_ms: job_status.updated_at_ms,
      elapsed_ms: job_status.elapsed_ms,
      message: job_status.message,
      op: job_status.op,
      vectors: state.tree.count,
      dim: state.tree.dim,
      k: state.k
    })
  end

  defp fetch_job_error(job) do
    case FP.Job.result(job) do
      {:error, reason} -> inspect(reason)
      _ -> "job_error"
    end
  end

  defp elapsed_since(nil), do: 0
  defp elapsed_since(started_at_ms) do
    max(System.monotonic_time(:millisecond) - started_at_ms, 0)
  end

  defp start_kmeans_job(state, data_bin) do
    try do
      job =
        FP.Job.start_kmeans(
          data_bin,
          state.tree.count,
          state.tree.dim,
          state.k,
          state.max_iter,
          state.tol,
          state.seed
        )

      Progress.report(%{
        status: :queued,
        phase: :init,
        iter: 0,
        max_iter: state.max_iter,
        started_at_ms: state.started_at_ms,
        updated_at_ms: System.monotonic_time(:millisecond),
        message: "queued",
        op: :fp_kmeans_f64,
        vectors: state.tree.count,
        dim: state.tree.dim,
        k: state.k
      })

      Process.send_after(self(), :poll, @poll_interval_ms)

      {:noreply, %{state | status: :queued, job: job}}
    rescue
      e ->
        Progress.report(%{
          status: :error,
          phase: :init,
          message: "job_start_failed",
          error: Exception.message(e),
          updated_at_ms: System.monotonic_time(:millisecond),
          op: :fp_kmeans_f64,
          vectors: state.tree.count,
          dim: state.tree.dim,
          k: state.k
        })

        {:noreply, %{status: :idle, job: nil}}
    end
  end
end
