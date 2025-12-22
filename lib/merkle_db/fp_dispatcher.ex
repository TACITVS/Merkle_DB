defmodule MerkleDb.FPDispatcher do
  @moduledoc false

  use GenServer
  alias MerkleDb.{ASM, FPManifest}

  @default_call_timeout :infinity
  @default_history_limit 500

  # --- Supervisor ---

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @impl true
  def init(_) do
    {:ok,
     %{
       queue: :queue.new(),
       queued: %{},
       running: %{},
       running_refs: %{},
       waiters: %{},
       results: %{},
       result_order: :queue.new(),
       next_job_id: 1,
       max_history: @default_history_limit,
       in_flight: 0,
       total_submitted: 0,
       total_completed: 0,
       total_failed: 0,
       total_canceled: 0,
       last_error: nil,
       last_error_at: nil,
       max_concurrency: System.schedulers_online(),
       default_call_timeout: @default_call_timeout
     }}
  end

  # --- Public API ---

  def status do
    case Process.whereis(__MODULE__) do
      nil -> default_status()
      _ -> GenServer.call(__MODULE__, :status, 200)
    end
  end

  def configure(opts) when is_list(opts) do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, {:configure, opts}, 200)
    end
  end

  def call(name, args, opts \\ []) do
    case dispatch(name, args, Keyword.put(opts, :await, true)) do
      {:ok, result} ->
        result

      {:error, {:exception, error, stack}} ->
        reraise error, stack

      {:error, {kind, reason, stack}} ->
        :erlang.raise(kind, reason, stack)

      {:error, reason} ->
        raise RuntimeError, "FP_ASM job failed: #{inspect(reason)}"
    end
  end

  def async(name, args, opts \\ []) do
    submit(name, args, opts)
    |> case do
      {:ok, job_id} -> job_id
      {:error, reason} -> raise RuntimeError, "FP_ASM job rejected: #{inspect(reason)}"
    end
  end

  def submit(name, args, opts \\ []) do
    dispatch(name, args, Keyword.put(opts, :await, false))
  end

  def await(job_id, timeout \\ @default_call_timeout) do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, {:await, job_id}, timeout)
    end
  end

  def result(job_id) do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, {:result, job_id}, 200)
    end
  end

  def cancel(job_id) do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, {:cancel, job_id}, 200)
    end
  end

  def cancel_all do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, :cancel_all, 200)
    end
  end

  def flush_queue do
    case Process.whereis(__MODULE__) do
      nil -> {:error, :not_running}
      _ -> GenServer.call(__MODULE__, :flush_queue, 200)
    end
  end

  def dispatch_many(jobs, opts \\ []) when is_list(jobs) do
    timeout = Keyword.get(opts, :timeout, @default_call_timeout)

    job_ids =
      Enum.map(jobs, fn job ->
        {name, args} = normalize_job(job)
        submit(name, args, opts)
      end)

    Enum.map(job_ids, fn
      {:ok, job_id} -> await(job_id, timeout)
      {:error, reason} -> {:error, reason}
    end)
  end

  # --- GenServer Callbacks ---

  @impl true
  def handle_call(:status, _from, state) do
    {:reply, status_snapshot(state), state}
  end

  @impl true
  def handle_call({:configure, opts}, _from, state) do
    updated = state |> apply_config(opts)
    {:reply, :ok, updated}
  end

  @impl true
  def handle_call({:submit, job, await?}, from, state) do
    {job_id, state} = enqueue_job(state, job)
    state = maybe_add_waiter(state, job_id, await?, from)
    state = dispatch_ready(state)

    if await? do
      {:noreply, state}
    else
      {:reply, {:ok, job_id}, state}
    end
  end

  @impl true
  def handle_call({:await, job_id}, from, state) do
    case Map.fetch(state.results, job_id) do
      {:ok, result} ->
        {:reply, result, state}

      :error ->
        if Map.has_key?(state.queued, job_id) or Map.has_key?(state.running, job_id) do
          {:noreply, add_waiter(state, job_id, from)}
        else
          {:reply, {:error, :unknown_job}, state}
        end
    end
  end

  @impl true
  def handle_call({:result, job_id}, _from, state) do
    reply =
      cond do
        Map.has_key?(state.results, job_id) -> Map.fetch!(state.results, job_id)
        Map.has_key?(state.running, job_id) -> {:error, :running}
        Map.has_key?(state.queued, job_id) -> {:error, :queued}
        true -> {:error, :unknown_job}
      end

    {:reply, reply, state}
  end

  @impl true
  def handle_call({:cancel, job_id}, _from, state) do
    {reply, updated} = cancel_job(state, job_id)
    {:reply, reply, dispatch_ready(updated)}
  end

  @impl true
  def handle_call(:cancel_all, _from, state) do
    {state, canceled_queue} = cancel_all_queued(state)
    {state, canceled_running} = cancel_all_running(state)

    reply = %{canceled_queue: canceled_queue, canceled_running: canceled_running}
    {:reply, reply, dispatch_ready(state)}
  end

  @impl true
  def handle_call(:flush_queue, _from, state) do
    {state, flushed} = flush_queued(state)
    {:reply, flushed, state}
  end

  @impl true
  def handle_cast({:track_sync_start, _entry}, state) do
    {:noreply,
     %{state | in_flight: state.in_flight + 1, total_submitted: state.total_submitted + 1}}
  end

  @impl true
  def handle_cast({:track_sync_finish, result}, state) do
    {:noreply, handle_finish(state, result)}
  end

  @impl true
  def handle_info({:job_done, job_id, result}, state) do
    case Map.fetch(state.running, job_id) do
      {:ok, running} ->
        state =
          state
          |> drop_running(job_id, running.ref)
          |> finish_job(job_id, result)

        {:noreply, dispatch_ready(state)}

      :error ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Map.fetch(state.running_refs, ref) do
      {:ok, job_id} ->
        result = {:error, {:exit, reason}}

        state =
          state
          |> drop_running(job_id, ref)
          |> finish_job(job_id, result)

        {:noreply, dispatch_ready(state)}

      :error ->
        {:noreply, state}
    end
  end

  # --- Dispatch ---

  defp dispatch(name, args, opts) do
    name_atom = normalize_name!(name)
    args_list = normalize_args!(args)
    module = Keyword.get(opts, :module, ASM)
    entry = resolve_entry!(name_atom, length(args_list), module, opts)
    mode = normalize_mode(Keyword.get(opts, :mode, entry.mode))
    await = Keyword.get(opts, :await, true)
    call_timeout = Keyword.get(opts, :timeout, default_call_timeout())

    cond do
      Keyword.get(opts, :force_sync, false) ->
        {:ok, execute_sync(module, name_atom, args_list, entry)}

      mode == "sync" ->
        {:ok, execute_sync(module, name_atom, args_list, entry)}

      Process.whereis(__MODULE__) == nil ->
        if await do
          {:ok, execute_sync(module, name_atom, args_list, entry)}
        else
          {:error, :not_running}
        end

      task_supervisor(opts) == nil ->
        if await do
          {:ok, execute_sync(module, name_atom, args_list, entry)}
        else
          {:error, :no_supervisor}
        end

      true ->
        job = build_job(name_atom, module, args_list, entry, opts)
        GenServer.call(__MODULE__, {:submit, job, await}, call_timeout)
    end
  end

  defp build_job(name_atom, module, args, entry, opts) do
    %{
      id: nil,
      name: name_atom,
      module: module,
      args: args,
      entry: entry,
      mode: normalize_mode(Keyword.get(opts, :mode, entry.mode)),
      submitted_at_ms: System.monotonic_time(:millisecond),
      timeout_ms: Keyword.get(opts, :job_timeout_ms, 0),
      supervisor: task_supervisor(opts)
    }
  end

  defp execute_sync(module, name_atom, args, entry) do
    track_sync_start(entry)

    try do
      result = apply(module, name_atom, args)
      track_sync_finish({:ok, result})
      result
    rescue
      e ->
        track_sync_finish({:error, {:exception, e, __STACKTRACE__}})
        reraise e, __STACKTRACE__
    catch
      kind, reason ->
        track_sync_finish({:error, {kind, reason, __STACKTRACE__}})
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  defp start_job(state, job) do
    parent = self()

    {:ok, pid} =
      Task.Supervisor.start_child(job.supervisor, fn ->
        result =
          try do
            {:ok, apply(job.module, job.name, job.args)}
          rescue
            e -> {:error, {:exception, e, __STACKTRACE__}}
          catch
            kind, reason -> {:error, {kind, reason, __STACKTRACE__}}
          end

        send(parent, {:job_done, job.id, result})
      end)

    ref = Process.monitor(pid)

    running = Map.put(state.running, job.id, %{pid: pid, ref: ref})
    running_refs = Map.put(state.running_refs, ref, job.id)

    %{
      state
      | running: running,
        running_refs: running_refs,
        in_flight: state.in_flight + 1
    }
  end

  # --- Queue Management ---

  defp enqueue_job(state, job) do
    job_id = state.next_job_id
    job = %{job | id: job_id}

    queue = :queue.in(job_id, state.queue)
    queued = Map.put(state.queued, job_id, job)

    updated = %{
      state
      | queue: queue,
        queued: queued,
        total_submitted: state.total_submitted + 1,
        next_job_id: job_id + 1
    }

    {job_id, updated}
  end

  defp dispatch_ready(state) do
    cond do
      map_size(state.running) >= state.max_concurrency -> state
      :queue.is_empty(state.queue) -> state
      true ->
        case pop_next_job(state) do
          {:ok, job, state} ->
            state
            |> start_job(job)
            |> dispatch_ready()

          {:empty, state} ->
            state
        end
    end
  end

  defp pop_next_job(state) do
    case :queue.out(state.queue) do
      {:empty, _} ->
        {:empty, state}

      {{:value, job_id}, queue} ->
        case Map.pop(state.queued, job_id) do
          {nil, queued} ->
            pop_next_job(%{state | queue: queue, queued: queued})

          {job, queued} ->
            {:ok, job, %{state | queue: queue, queued: queued}}
        end
    end
  end

  defp finish_job(state, job_id, result) do
    state
    |> handle_finish(result)
    |> store_result(job_id, result)
    |> reply_waiters(job_id, result)
  end

  defp handle_finish(state, {:ok, _}) do
    %{state | in_flight: max(state.in_flight - 1, 0), total_completed: state.total_completed + 1}
  end

  defp handle_finish(state, {:error, reason}) do
    %{
      state
      | in_flight: max(state.in_flight - 1, 0),
        total_failed: state.total_failed + 1,
        last_error: reason,
        last_error_at: DateTime.to_iso8601(DateTime.utc_now() |> DateTime.truncate(:second))
    }
  end

  defp drop_running(state, job_id, ref) do
    %{
      state
      | running: Map.delete(state.running, job_id),
        running_refs: Map.delete(state.running_refs, ref)
    }
  end

  defp store_result(state, job_id, result) do
    results = Map.put(state.results, job_id, result)
    result_order = :queue.in(job_id, state.result_order)

    trim_results(%{state | results: results, result_order: result_order})
  end

  defp trim_results(state) do
    if map_size(state.results) <= state.max_history do
      state
    else
      case :queue.out(state.result_order) do
        {:empty, _} ->
          state

        {{:value, old_job_id}, order} ->
          trim_results(%{
            state
            | results: Map.delete(state.results, old_job_id),
              result_order: order
          })
      end
    end
  end

  # --- Waiters ---

  defp maybe_add_waiter(state, _job_id, false, _from), do: state
  defp maybe_add_waiter(state, job_id, true, from), do: add_waiter(state, job_id, from)

  defp add_waiter(state, job_id, from) do
    waiters = Map.update(state.waiters, job_id, [from], &[from | &1])
    %{state | waiters: waiters}
  end

  defp reply_waiters(state, job_id, result) do
    case Map.pop(state.waiters, job_id) do
      {nil, _} ->
        state

      {waiters, remaining} ->
        Enum.each(waiters, &GenServer.reply(&1, result))
        %{state | waiters: remaining}
    end
  end

  # --- Cancellation ---

  defp cancel_job(state, job_id) do
    cond do
      Map.has_key?(state.queued, job_id) ->
        {job, queued} = Map.pop(state.queued, job_id)
        state = %{state | queued: queued}
        state = register_canceled(state, job_id, :queued, job)
        {{:ok, :queued}, state}

      Map.has_key?(state.running, job_id) ->
        running = Map.fetch!(state.running, job_id)
        Process.exit(running.pid, :kill)
        state = drop_running(state, job_id, running.ref)
        state = %{state | in_flight: max(state.in_flight - 1, 0)}
        state = register_canceled(state, job_id, :running, nil)
        {{:ok, :running}, state}

      Map.has_key?(state.results, job_id) ->
        {{:error, :already_finished}, state}

      true ->
        {{:error, :unknown_job}, state}
    end
  end

  defp register_canceled(state, job_id, stage, _job) do
    result = {:error, {:canceled, stage}}

    state
    |> Map.update!(:total_canceled, &(&1 + 1))
    |> store_result(job_id, result)
    |> reply_waiters(job_id, result)
  end

  defp cancel_all_queued(state) do
    job_ids = Map.keys(state.queued)
    state = %{state | queued: %{}}

    state =
      Enum.reduce(job_ids, state, fn job_id, acc ->
        register_canceled(acc, job_id, :queued, nil)
      end)

    {state, length(job_ids)}
  end

  defp cancel_all_running(state) do
    running_ids = Map.keys(state.running)

    state =
      Enum.reduce(running_ids, state, fn job_id, acc ->
        {_, acc} = cancel_job(acc, job_id)
        acc
      end)

    {state, length(running_ids)}
  end

  defp flush_queued(state) do
    job_ids = Map.keys(state.queued)
    state = %{state | queued: %{}, queue: :queue.new()}

    state =
      Enum.reduce(job_ids, state, fn job_id, acc ->
        register_canceled(acc, job_id, :queued, nil)
      end)

    {state, length(job_ids)}
  end

  # --- Metrics Helpers ---

  defp track_sync_start(entry) do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, {:track_sync_start, entry})
    else
      :ok
    end
  end

  defp track_sync_finish(result) do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, {:track_sync_finish, result})
    else
      :ok
    end
  end

  defp status_snapshot(state) do
    %{
      in_flight: state.in_flight,
      queue_depth: map_size(state.queued),
      total_submitted: state.total_submitted,
      total_completed: state.total_completed,
      total_failed: state.total_failed,
      total_canceled: state.total_canceled,
      last_error: state.last_error,
      last_error_at: state.last_error_at,
      max_concurrency: state.max_concurrency,
      default_call_timeout: state.default_call_timeout
    }
  end

  defp default_status do
    %{
      in_flight: 0,
      queue_depth: 0,
      total_submitted: 0,
      total_completed: 0,
      total_failed: 0,
      total_canceled: 0,
      last_error: nil,
      last_error_at: nil,
      max_concurrency: System.schedulers_online(),
      default_call_timeout: @default_call_timeout
    }
  end

  defp apply_config(state, opts) do
    state
    |> maybe_put(:max_concurrency, opts)
    |> maybe_put(:default_call_timeout, opts)
    |> maybe_put(:max_history, opts)
  end

  defp maybe_put(state, key, opts) do
    case Keyword.fetch(opts, key) do
      {:ok, value} when is_integer(value) and value > 0 -> Map.put(state, key, value)
      {:ok, :infinity} when key == :default_call_timeout -> Map.put(state, key, :infinity)
      _ -> state
    end
  end

  defp default_call_timeout do
    case status() do
      %{default_call_timeout: timeout} -> timeout
      _ -> @default_call_timeout
    end
  end

  # --- Entry Resolution ---

  defp resolve_entry!(name_atom, arity, module, opts) do
    if module == ASM do
      entry =
        case FPManifest.lookup(name_atom, arity) do
          {:ok, entry} -> entry
          :error -> raise ArgumentError, "unknown FP_ASM function #{name_atom}/#{arity}"
        end

      if entry.allowed do
        mode_override = Keyword.get(opts, :mode, entry.mode)
        %{entry | mode: normalize_mode(mode_override)}
      else
        raise ArgumentError, "FP_ASM function excluded: #{name_atom}/#{arity}"
      end
    else
      mode =
        case Keyword.fetch(opts, :mode) do
          {:ok, value} -> normalize_mode(value)
          :error -> raise ArgumentError, "mode is required for custom module dispatch"
        end

      %{
        name: Atom.to_string(name_atom),
        arity: arity,
        category: Keyword.get(opts, :category, "custom"),
        mode: mode,
        allowed: true
      }
    end
  end

  defp normalize_name!(name) when is_atom(name), do: name

  defp normalize_name!(name) when is_binary(name) do
    String.to_existing_atom(name)
  rescue
    ArgumentError ->
      raise ArgumentError, "unknown FP_ASM function #{name}"
  end

  defp normalize_name!(_), do: raise(ArgumentError, "FP_ASM function name must be an atom or string")

  defp normalize_args!(args) when is_list(args), do: args
  defp normalize_args!(_), do: raise(ArgumentError, "FP_ASM args must be a list")

  defp normalize_mode(mode) when is_atom(mode), do: Atom.to_string(mode)
  defp normalize_mode(mode) when is_binary(mode), do: mode
  defp normalize_mode(_), do: "sync"

  defp normalize_job({name, args}), do: {name, args}
  defp normalize_job(%{name: name, args: args}), do: {name, args}
  defp normalize_job(_), do: raise(ArgumentError, "job must be {name, args} or %{name: name, args: args}")

  defp task_supervisor(opts) do
    case Keyword.get(opts, :supervisor) do
      nil -> Process.whereis(MerkleDb.TaskSupervisor)
      name when is_atom(name) -> Process.whereis(name)
      pid when is_pid(pid) -> pid
      _ -> nil
    end
  end
end
