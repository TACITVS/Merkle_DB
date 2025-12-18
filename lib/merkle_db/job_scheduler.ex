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
    tree = KV.snapshot()
    if tree.count == 0, do: throw("Empty DB")
    
    queue = Map.keys(tree.keys) |> Enum.shuffle()
    
    new_state = %{
      status: :running, 
      queue: queue, 
      visited: MapSet.new(), 
      topics: [],
      total: length(queue)
    }
    
    send(self(), :tick)
    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:pause, state), do: {:noreply, %{state | status: :paused}}
  
  @impl true
  def handle_cast(:resume, state) do
    send(self(), :tick)
    {:noreply, %{state | status: :running}}
  end

  @impl true
  def handle_cast(:stop, state), do: {:noreply, %{state | status: :idle}}

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
    scanned = state.total - length(state.queue)
    percent = if state.total == 0, do: 0, else: Float.round((scanned / state.total) * 100, 1)
    
    response = %{
      status: state.status,
      topics: state.topics,
      percent: percent,
      found_count: length(state.topics)
    }
    {:reply, response, state}
  end

  # --- THE LOOP ---
  @impl true
  def handle_info(:tick, state) do
    cond do
      state.status != :running -> {:noreply, state}
      state.queue == [] -> {:noreply, %{state | status: :done}}
      true ->
        # 1. Pop next candidate
        [idx | rest] = state.queue
        tree = KV.snapshot()

        # 2. Analyze (Pure Function)
        # 3. Update State based on result
        {final_topics, final_visited} = case Cluster.analyze_step(tree, idx, state.visited) do
           {:found, topic, v} -> {[topic | state.topics], v}
           {:skip, v} -> {state.topics, v}
        end

        # 4. Schedule next tick
        send(self(), :tick)
        
        {:noreply, %{state | queue: rest, visited: final_visited, topics: final_topics}}
    end
  end
end