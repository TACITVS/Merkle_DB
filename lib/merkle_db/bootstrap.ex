defmodule MerkleDb.Bootstrap do
  use GenServer

  alias MerkleDb.{IndexBuilder, KV, Persistence, Progress, TextEmbedding, TextStore, Tree}

  @default_state %{
    status: :idle,
    phase: :idle,
    percent: 0.0,
    message: "",
    error: nil,
    started_at_ms: nil,
    updated_at_ms: nil,
    elapsed_ms: 0,
    processed: 0,
    total: 0,
    mode: :auto,
    task: nil
  }

  @default_opts [
    mode: :auto,
    build_index: true,
    save_snapshot: true,
    batch_size: 200,
    concurrency: nil,
    poll_interval_ms: 500,
    min_vectors: 10,
    max_iter: 100,
    tol: 1.0e-4,
    seed: 42
  ]

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, @default_state, name: __MODULE__)
  end

  def start(opts \\ []) do
    GenServer.call(__MODULE__, {:start, opts})
  end

  def save_snapshot(opts \\ []) do
    GenServer.call(__MODULE__, {:save_snapshot, opts})
  end

  def clear_snapshot do
    GenServer.call(__MODULE__, :clear_snapshot)
  end

  def cancel do
    GenServer.call(__MODULE__, :cancel)
  end

  def status(opts \\ []) do
    GenServer.call(__MODULE__, {:status, opts})
  end

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_call({:start, opts}, _from, state) do
    start_job(opts, state)
  end

  @impl true
  def handle_call({:save_snapshot, opts}, _from, state) do
    opts = Keyword.put(opts, :mode, :save_snapshot)
    start_job(opts, state)
  end

  @impl true
  def handle_call(:clear_snapshot, _from, state) do
    case state.status do
      :running ->
        {:reply, {:error, :busy}, state}

      _ ->
        _ = Persistence.delete()
        {:reply, :ok, state}
    end
  end

  @impl true
  def handle_call(:cancel, _from, state) do
    case state.task do
      nil ->
        {:reply, {:error, :not_running}, state}

      task ->
        _ = Task.shutdown(task, :brutal_kill)
        updated =
          state
          |> Map.merge(%{
            status: :cancelled,
            phase: :cancelled,
            message: "cancelled",
            updated_at_ms: System.monotonic_time(:millisecond),
            task: nil
          })
          |> normalize()

        {:reply, :ok, updated}
    end
  end

  @impl true
  def handle_call({:status, opts}, _from, state) do
    snapshot = Persistence.snapshot_info()
    tree_stats = Keyword.get(opts, :tree_stats) || (KV.snapshot() |> Tree.stats())
    text_count = Keyword.get(opts, :text_count, TextStore.count())
    recommendation = recommend(tree_stats, snapshot, text_count)

    response =
      state
      |> Map.drop([:task])
      |> Map.merge(%{
        snapshot: snapshot,
        tree: tree_stats,
        text_count: text_count,
        recommendation: recommendation
      })
      |> normalize()

    {:reply, response, state}
  end

  @impl true
  def handle_info({:bootstrap_update, update}, state) do
    new_state =
      state
      |> Map.merge(update)
      |> normalize()

    {:noreply, new_state}
  end

  @impl true
  def handle_info({ref, result}, state) when not is_nil(state.task) and ref == state.task.ref do
    updated =
      case result do
        {:ok, _summary} ->
          final_message =
            if state.message in [nil, "", "starting"] do
              "complete"
            else
              state.message
            end

          state
          |> Map.merge(%{
            status: :done,
            phase: :complete,
            percent: 100.0,
            message: final_message,
            error: nil,
            updated_at_ms: System.monotonic_time(:millisecond),
            task: nil
          })

        {:error, reason} ->
          state
          |> Map.merge(%{
            status: :error,
            phase: :error,
            message: "failed",
            error: format_error(reason),
            updated_at_ms: System.monotonic_time(:millisecond),
            task: nil
          })
      end

    {:noreply, normalize(updated)}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state)
      when not is_nil(state.task) and ref == state.task.ref do
    updated =
      if state.status == :running do
        state
        |> Map.merge(%{
          status: :error,
          phase: :error,
          message: "failed",
          error: format_error(reason),
          updated_at_ms: System.monotonic_time(:millisecond),
          task: nil
        })
      else
        state
      end

    {:noreply, normalize(updated)}
  end

  defp start_job(opts, state) do
    if state.status in [:running, :preparing, :loading, :rebuilding, :building_index, :saving] do
      {:reply, {:error, :already_running}, state}
    else
      merged_opts = Keyword.merge(@default_opts, opts)
      mode = Keyword.get(merged_opts, :mode, :auto)
      started_at_ms = System.monotonic_time(:millisecond)

      task =
        Task.Supervisor.async_nolink(MerkleDb.TaskSupervisor, fn ->
          run_bootstrap(mode, merged_opts, self())
        end)

      new_state =
        state
        |> Map.merge(%{
          status: :running,
          phase: :init,
          percent: 0.0,
          message: "starting",
          error: nil,
          started_at_ms: started_at_ms,
          updated_at_ms: started_at_ms,
          processed: 0,
          total: 0,
          mode: mode,
          task: task
        })
        |> normalize()

      {:reply, {:ok, %{status: :running, mode: mode}}, new_state}
    end
  end

  defp run_bootstrap(mode, opts, parent) do
    report(parent, %{phase: :init, message: "starting"})

    tree = KV.snapshot()
    stats = Tree.stats(tree)
    snapshot = Persistence.snapshot_info()
    text_count = TextStore.count()

    case mode do
      :auto ->
        run_auto(stats, snapshot, text_count, opts, parent)

      :load_snapshot ->
        with {:ok, tree} <- load_snapshot(parent),
             {:ok, tree} <- maybe_build_index(tree, opts, parent),
             {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :load_snapshot}}
        end

      :rebuild ->
        with {:ok, tree} <- rebuild_from_texts(text_count, opts, parent),
             :ok <- KV.set_tree(tree),
             {:ok, tree} <- maybe_build_index(tree, opts, parent),
             {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :rebuild}}
        end

      :build_index ->
        with {:ok, tree} <- maybe_build_index(tree, opts, parent),
             {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :build_index}}
        end

      :save_snapshot ->
        with {:ok, meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :save_snapshot, snapshot: meta}}
        end

      _ ->
        {:error, :invalid_mode}
    end
  end

  defp run_auto(stats, snapshot, text_count, opts, parent) do
    cond do
      stats.count > 0 ->
        report(parent, %{phase: :preparing, message: "using_existing_tree", percent: 0.0})

        with {:ok, tree} <- maybe_build_index(KV.snapshot(), opts, parent),
             {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :reuse_existing}}
        end

      snapshot.exists ->
        report(parent, %{phase: :loading, message: "loading_snapshot", percent: 0.0})

        with {:ok, tree} <- load_snapshot(parent),
             {:ok, tree} <- maybe_build_index(tree, opts, parent),
             {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
          {:ok, %{action: :loaded_snapshot}}
        else
          {:error, _} = error ->
            if text_count > 0 do
              report(parent, %{phase: :rebuilding, message: "snapshot_failed_rebuilding", percent: 0.0})
              run_rebuild(text_count, opts, parent)
            else
              error
            end
        end

      text_count > 0 ->
        report(parent, %{phase: :rebuilding, message: "rebuilding_from_text", percent: 0.0})
        run_rebuild(text_count, opts, parent)

      true ->
        report(parent, %{phase: :idle, message: "no_data_ingest_required", percent: 0.0})
        {:ok, %{action: :no_data}}
    end
  end

  defp run_rebuild(text_count, opts, parent) do
    with {:ok, tree} <- rebuild_from_texts(text_count, opts, parent),
         :ok <- KV.set_tree(tree),
         {:ok, tree} <- maybe_build_index(tree, opts, parent),
         {:ok, _meta} <- maybe_save_snapshot(tree, opts, parent) do
      {:ok, %{action: :rebuild}}
    end
  end

  defp load_snapshot(parent) do
    report(parent, %{phase: :loading, message: "loading_snapshot", percent: 0.0})

    case Persistence.load() do
      {:ok, %{tree: tree}} ->
        :ok = KV.set_tree(tree)
        report(parent, %{phase: :loading, message: "snapshot_loaded", percent: 100.0})
        {:ok, tree}

      {:error, reason} ->
        report(parent, %{phase: :loading, message: "snapshot_failed", error: format_error(reason)})
        {:error, reason}
    end
  end

  defp rebuild_from_texts(text_count, opts, parent) do
    if text_count <= 0 do
      {:error, :no_text_data}
    else
      batch_size = Keyword.get(opts, :batch_size, 200)
      concurrency = Keyword.get(opts, :concurrency) || System.schedulers_online()
      all_texts = TextStore.get_all()

      {tree, processed} =
        all_texts
        |> Enum.chunk_every(batch_size)
        |> Enum.reduce({Tree.new(), 0}, fn batch, {tree_acc, done} ->
          embeddings = embed_batch(batch, concurrency)
          new_tree = Tree.insert_batch(tree_acc, embeddings)
          next_done = done + length(embeddings)
          percent = progress_percent(next_done, text_count)

          report(parent, %{
            phase: :rebuilding,
            message: "embedding_text",
            percent: percent,
            processed: next_done,
            total: text_count
          })

          {new_tree, next_done}
        end)

      report(parent, %{
        phase: :rebuilding,
        message: "embedding_complete",
        percent: 100.0,
        processed: processed,
        total: text_count
      })

      {:ok, tree}
    end
  end

  defp embed_batch(batch, concurrency) do
    batch
    |> Task.async_stream(
      fn {key, text} -> {key, TextEmbedding.embed(text)} end,
      max_concurrency: concurrency,
      ordered: false,
      timeout: :infinity
    )
    |> Enum.map(fn
      {:ok, item} -> item
      {:exit, reason} -> raise "embedding_failed: #{inspect(reason)}"
    end)
  end

  defp maybe_build_index(tree, opts, parent) do
    if Keyword.get(opts, :build_index, true) && tree.centroids == nil do
      if tree.count < Keyword.get(opts, :min_vectors, 10) do
        report(parent, %{phase: :building_index, message: "not_enough_vectors", percent: 0.0})
        {:ok, tree}
      else
        k = max(10, trunc(:math.sqrt(tree.count)))
        report(parent, %{phase: :building_index, message: "starting_index", percent: 0.0})

        case IndexBuilder.start_build(k,
               max_iter: Keyword.get(opts, :max_iter, 100),
               tol: Keyword.get(opts, :tol, 1.0e-4),
               seed: Keyword.get(opts, :seed, 42),
               auto_snapshot: false
             ) do
          {:ok, _} ->
            case wait_for_index(parent, Keyword.get(opts, :poll_interval_ms, 500)) do
              :ok -> {:ok, KV.snapshot()}
              {:error, reason} -> {:error, reason}
            end

          {:error, :already_running} ->
            case wait_for_index(parent, Keyword.get(opts, :poll_interval_ms, 500)) do
              :ok -> {:ok, KV.snapshot()}
              {:error, reason} -> {:error, reason}
            end

          {:error, reason} ->
            report(parent, %{phase: :building_index, message: "index_start_failed", error: format_error(reason)})
            {:error, reason}
        end
      end
    else
      {:ok, tree}
    end
  end

  defp wait_for_index(parent, poll_interval_ms) do
    status = Progress.get_status()

    report(parent, %{
      phase: :building_index,
      message: status.message || "index_building",
      percent: status.percent || 0.0
    })

    case status.status do
      :done ->
        report(parent, %{phase: :building_index, message: "index_complete", percent: 100.0})
        :ok

      :error ->
        report(parent, %{phase: :building_index, message: "index_failed", error: status.error})
        {:error, :index_failed}

      :stale ->
        report(parent, %{phase: :building_index, message: "index_stale"})
        {:error, :index_stale}

      :cancelled ->
        report(parent, %{phase: :building_index, message: "index_cancelled"})
        {:error, :index_cancelled}

      _ ->
        if status.status == :idle do
          if KV.snapshot().centroids != nil do
            report(parent, %{phase: :building_index, message: "index_complete", percent: 100.0})
            :ok
          else
            report(parent, %{phase: :building_index, message: "index_idle"})
            {:error, :index_idle}
          end
        else
          Process.sleep(poll_interval_ms)
          wait_for_index(parent, poll_interval_ms)
        end
    end
  end

  defp maybe_save_snapshot(tree, opts, parent) do
    if Keyword.get(opts, :save_snapshot, true) && tree.count > 0 do
      report(parent, %{phase: :saving, message: "saving_snapshot", percent: 0.0})

      case Persistence.save(tree, label: Keyword.get(opts, :label, "bootstrap")) do
        {:ok, meta} ->
          report(parent, %{phase: :saving, message: "snapshot_saved", percent: 100.0})
          {:ok, meta}

        {:error, reason} ->
          report(parent, %{phase: :saving, message: "snapshot_failed", error: format_error(reason)})
          {:error, reason}
      end
    else
      {:ok, %{skipped: true}}
    end
  end

  defp recommend(stats, snapshot, text_count) do
    cond do
      stats.count > 0 and stats.has_ivf_index == false -> :build_index
      stats.count > 0 and !snapshot.exists -> :save_snapshot
      stats.count > 0 -> :none
      snapshot.exists -> :load_snapshot
      text_count > 0 -> :rebuild
      true -> :ingest
    end
  end

  defp report(parent, update) do
    send(parent, {:bootstrap_update, Map.put(update, :updated_at_ms, System.monotonic_time(:millisecond))})
  end

  defp normalize(state) do
    started_at_ms = Map.get(state, :started_at_ms)
    updated_at_ms = Map.get(state, :updated_at_ms, started_at_ms)

    elapsed_ms =
      if started_at_ms && updated_at_ms do
        max(updated_at_ms - started_at_ms, 0)
      else
        Map.get(state, :elapsed_ms, 0)
      end

    Map.put(state, :elapsed_ms, elapsed_ms)
  end

  defp progress_percent(done, total) do
    if total > 0 do
      Float.round(done / total * 100.0, 1)
    else
      0.0
    end
  end

  defp format_error(reason) when is_binary(reason), do: reason
  defp format_error(reason), do: inspect(reason)
end
