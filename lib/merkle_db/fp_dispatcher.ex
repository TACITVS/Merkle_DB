defmodule MerkleDb.FPDispatcher do
  @moduledoc false

  use GenServer
  alias MerkleDb.{ASM, FPManifest}

  @default_timeout_ms 30_000

  # --- Supervisor ---

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @impl true
  def init(_) do
    {:ok,
     %{
       in_flight: 0,
       total_submitted: 0,
       total_completed: 0,
       total_failed: 0,
       last_error: nil,
       last_error_at: nil,
       max_concurrency: System.schedulers_online(),
       default_timeout_ms: @default_timeout_ms
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
    dispatch(name, args, Keyword.put(opts, :await, true))
  end

  def async(name, args, opts \\ []) do
    dispatch(name, args, Keyword.put(opts, :await, false))
  end

  def dispatch_many(jobs, opts \\ []) when is_list(jobs) do
    supervisor = task_supervisor(opts)
    timeout = Keyword.get(opts, :timeout, default_timeout_ms())
    max_concurrency = Keyword.get(opts, :max_concurrency, default_max_concurrency())
    ordered = Keyword.get(opts, :ordered, false)

    if supervisor do
      Task.Supervisor.async_stream_nolink(
        supervisor,
        jobs,
        fn job ->
          {name, args} = normalize_job(job)
          dispatch(name, args, Keyword.merge(opts, await: true, force_sync: true))
        end,
        max_concurrency: max_concurrency,
        ordered: ordered,
        timeout: timeout
      )
      |> Enum.to_list()
    else
      Enum.map(jobs, fn job ->
        try do
          {name, args} = normalize_job(job)
          {:ok, dispatch(name, args, Keyword.merge(opts, await: true, force_sync: true))}
        rescue
          e -> {:exit, e}
        end
      end)
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def handle_call(:status, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:configure, opts}, _from, state) do
    updated = state |> apply_config(opts)
    {:reply, :ok, updated}
  end

  @impl true
  def handle_cast({:track_start, _entry}, state) do
    {:noreply,
     %{state | in_flight: state.in_flight + 1, total_submitted: state.total_submitted + 1}}
  end

  @impl true
  def handle_cast({:track_finish, :ok}, state) do
    {:noreply,
     %{
       state
       | in_flight: max(state.in_flight - 1, 0),
         total_completed: state.total_completed + 1
     }}
  end

  @impl true
  def handle_cast({:track_finish, {:error, reason}}, state) do
    {:noreply,
     %{
       state
       | in_flight: max(state.in_flight - 1, 0),
         total_failed: state.total_failed + 1,
         last_error: reason,
         last_error_at: DateTime.to_iso8601(DateTime.utc_now() |> DateTime.truncate(:second))
     }}
  end

  # --- Dispatch ---

  defp dispatch(name, args, opts) do
    name_atom = normalize_name!(name)
    args_list = normalize_args!(args)
    entry = resolve_entry!(name_atom, length(args_list), opts)
    mode = normalize_mode(Keyword.get(opts, :mode, entry.mode))
    module = Keyword.get(opts, :module, ASM)
    timeout = Keyword.get(opts, :timeout, default_timeout_ms())
    await = Keyword.get(opts, :await, true)
    force_sync = Keyword.get(opts, :force_sync, false)
    supervisor = task_supervisor(opts)

    cond do
      force_sync -> execute_sync(module, name_atom, args_list, entry)
      mode == "sync" -> execute_sync(module, name_atom, args_list, entry)
      supervisor == nil -> execute_sync(module, name_atom, args_list, entry)
      true ->
        track_start(entry)

        task =
          Task.Supervisor.async_nolink(supervisor, fn ->
            execute_task(module, name_atom, args_list, entry)
          end)

        if await do
          Task.await(task, timeout)
        else
          task
        end
    end
  end

  defp execute_sync(module, name_atom, args, entry) do
    track_start(entry)

    try do
      result = apply(module, name_atom, args)
      track_finish(:ok)
      result
    rescue
      e ->
        track_finish({:error, Exception.message(e)})
        reraise e, __STACKTRACE__
    catch
      kind, reason ->
        track_finish({:error, {kind, reason}})
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  defp execute_task(module, name_atom, args, _entry) do
    try do
      result = apply(module, name_atom, args)
      track_finish(:ok)
      result
    rescue
      e ->
        track_finish({:error, Exception.message(e)})
        reraise e, __STACKTRACE__
    catch
      kind, reason ->
        track_finish({:error, {kind, reason}})
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  # --- Metrics ---

  defp track_start(entry) do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, {:track_start, entry})
    else
      :ok
    end
  end

  defp track_finish(result) do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, {:track_finish, result})
    else
      :ok
    end
  end

  defp default_status do
    %{
      in_flight: 0,
      total_submitted: 0,
      total_completed: 0,
      total_failed: 0,
      last_error: nil,
      last_error_at: nil,
      max_concurrency: System.schedulers_online(),
      default_timeout_ms: @default_timeout_ms
    }
  end

  defp apply_config(state, opts) do
    state
    |> maybe_put(:max_concurrency, opts)
    |> maybe_put(:default_timeout_ms, opts)
  end

  defp maybe_put(state, key, opts) do
    case Keyword.fetch(opts, key) do
      {:ok, value} when is_integer(value) and value > 0 -> Map.put(state, key, value)
      _ -> state
    end
  end

  defp default_max_concurrency do
    case status() do
      %{max_concurrency: max} when is_integer(max) and max > 0 -> max
      _ -> System.schedulers_online()
    end
  end

  defp default_timeout_ms do
    case status() do
      %{default_timeout_ms: timeout} when is_integer(timeout) and timeout > 0 -> timeout
      _ -> @default_timeout_ms
    end
  end

  # --- Entry Resolution ---

  defp resolve_entry!(name_atom, arity, opts) do
    case Keyword.fetch(opts, :mode) do
      {:ok, mode} ->
        %{
          name: Atom.to_string(name_atom),
          arity: arity,
          category: Keyword.get(opts, :category, "custom"),
          mode: normalize_mode(mode),
          allowed: true
        }

      :error ->
        case FPManifest.lookup(name_atom, arity) do
          {:ok, entry} -> entry
          :error -> raise ArgumentError, "unknown FP_ASM function #{name_atom}/#{arity}"
        end
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
