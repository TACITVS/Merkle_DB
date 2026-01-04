defmodule MerkleDb.JobScheduler do
  use GenServer
  alias MerkleDb.{Cluster, KV}

  # --- API ---
  def start_link(_), do: GenServer.start_link(__MODULE__, :idle, name: __MODULE__)

  def start_job, do: GenServer.cast(__MODULE__, :start)
  def pause_job, do: GenServer.cast(__MODULE__, :pause)
  def resume_job, do: GenServer.cast(__MODULE__, :resume)
  def stop_job, do: GenServer.cast(__MODULE__, :stop)
  
  def save_state, do: GenServer.call(__MODULE__, :save_to_disk)
  def load_state, do: GenServer.cast(__MODULE__, :load_from_disk)

  def get_status, do: GenServer.call(__MODULE__, :get_status)

  # --- SERVER ---
  
  @impl true
  def init(_) do
    {:ok, %{status: :idle, queue: [], visited: MapSet.new(), topics: [], total: 0}}
  end

  @impl true
  def handle_cast(:start, _state) do
    # 1. Grab Immutable Snapshot (Thread-Safe)
    tree = KV.snapshot()
    if tree.count == 0, do: throw("Empty DB")
    
    queue = Map.keys(tree.keys) |> Enum.shuffle()
    
    # 2. Update state to indicate running
    new_state = %{
      status: :running, 
      queue: [], 
      visited: MapSet.new(), 
      topics: [],
      total: length(queue)
    }
    
    # 3. Spawn the Coordinator Task to keep the GenServer responsive.
    parent = self()
    Task.start_link(fn ->
      run_parallel_job(tree, queue, parent)
    end)

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:pause, state), do: {:noreply, %{state | status: :paused}}
  
  @impl true
  def handle_cast(:resume, state) do
    {:noreply, %{state | status: :running}}
  end

  @impl true
  def handle_cast(:stop, state), do: {:noreply, %{state | status: :idle}}

  @impl true
  def handle_cast({:job_complete, {topics, visited}}, state) do
    IO.puts("Parallel job complete. Found #{length(topics)} topics.")
    {:noreply, %{state | status: :done, topics: topics, visited: visited}}
  end

  @impl true
  def handle_cast(:load_from_disk, _state) do
    case File.read("job_state.bin") do
      {:ok, binary} -> 
        restored = :erlang.binary_to_term(binary)
        IO.puts("💾 Loaded Job: #{length(restored.topics)} topics found so far.")
        {:noreply, %{restored | status: :paused}}
      _ -> 
        IO.puts("⚠️ No saved job found.")
        {:noreply, %{status: :idle, queue: [], visited: MapSet.new(), topics: [], total: 0}}
    end
  end

  @impl true
  def handle_call(:save_to_disk, _from, state) do
    binary = :erlang.term_to_binary(state)
    File.write("job_state.bin", binary)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    percent =
      case state.status do
        :done -> 100.0
        _ -> 0.0
      end
    
    response = %{
      status: state.status,
      topics: state.topics,
      percent: percent,
      found_count: length(state.topics)
    }
    {:reply, response, state}
  end

  # --- THE PARALLEL PIPELINE ---
  defp run_parallel_job(tree, queue, parent_pid) do
    cores = System.schedulers_online()
    IO.puts("Starting parallel job on #{cores} schedulers.")

    results =
      queue
      |> Task.async_stream(fn idx ->
        Cluster.analyze_step(tree, idx, MapSet.new())
      end, max_concurrency: cores, ordered: false, timeout: :infinity)
      |> Enum.reduce({[], MapSet.new()}, fn
        {:ok, result}, {acc_topics, acc_visited} ->
          case result do
            {:found, topic, new_visited} ->
              {[topic | acc_topics], MapSet.union(acc_visited, new_visited)}
            {:skip, new_visited} ->
              {acc_topics, MapSet.union(acc_visited, new_visited)}
          end
        {:exit, reason}, acc ->
          IO.puts("Worker crashed: #{inspect(reason)}")
          acc
      end)

    GenServer.cast(parent_pid, {:job_complete, results})
  end
end
