defmodule MerkleDb.Web.Router do
  use Plug.Router
  alias MerkleDb.{Bootstrap, KV, Persistence, Query, TextEmbedding, TextStore, JobScheduler, Analytics, BenchmarkRunner, TelemetryAggregator, TextAnalytics, LoadGenerator, IndexBuilder}

  plug Plug.Static,
    at: "/",
    from: {:merkle_db, "priv/static"},
    only: ["index.html"]

  plug :match
  plug :dispatch

  # --- JOB CONTROLS ---

  post "/job/start" do
    tree = KV.snapshot()
    if tree.count > 0 do
      JobScheduler.start_job()
      send_resp(conn, 200, "Started")
    else
      send_resp(conn, 400, "Database is empty! Please click 'Ingest' first.")
    end
  end

  post "/job/pause" do
    JobScheduler.pause_job()
    send_resp(conn, 200, "Paused")
  end

  post "/job/resume" do
    JobScheduler.resume_job()
    send_resp(conn, 200, "Resumed")
  end

  post "/job/stop" do
    JobScheduler.stop_job()
    send_resp(conn, 200, "Stopped")
  end

  post "/job/save" do
    JobScheduler.save_state()
    send_resp(conn, 200, "Saved to Disk")
  end

  post "/job/load" do
    JobScheduler.load_state()
    send_resp(conn, 200, "Loaded from Disk")
  end

  get "/job/status" do
    if Process.whereis(JobScheduler) == nil, do: JobScheduler.start_link(nil)
    status = JobScheduler.get_status()
    topics_json = 
      status.topics 
      |> Enum.map(fn t -> "{\"label\": \"#{escape(t.label)}\", \"count\": #{t.count}}" end)
      |> Enum.join(",")

    json = """
    {
      "status": "#{status.status}",
      "percent": #{status.percent},
      "found_count": #{status.found_count},
      "topics": [#{topics_json}]
    }
    """
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  # --- STANDARD ENDPOINTS ---

  get "/" do
    if Process.whereis(JobScheduler) == nil, do: JobScheduler.start_link(nil)
    conn |> put_resp_content_type("text/html") |> send_file(200, Application.app_dir(:merkle_db, "priv/static/index.html"))
  end

  get "/favicon.ico" do
    send_resp(conn, 204, "")
  end

  post "/ingest" do
    case ensure_allowed(conn, :ingest) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        if Application.get_env(:merkle_db, :ingesting, false) do
          send_resp(conn, 429, "Busy")
        else
          path = "C:/Users/baian/AppData/Roaming/nltk_data/corpora/gutenberg/bible-kjv.txt"
          if File.exists?(path) do
            Application.put_env(:merkle_db, :ingesting, true)
            Task.start(fn ->
              try do
                ingest_bible(path)
              after
                Application.put_env(:merkle_db, :ingesting, false)
              end
            end)
            send_resp(conn, 202, "Started")
          else
            send_resp(conn, 404, "File not found")
          end
        end
    end
  end

  get "/analytics/summary" do
    try do
      tree = KV.snapshot()
      if tree.count > 0 and tree.dim > 0 do
        # Get stats for first 6 dimensions (0-5)
        stats = for i <- 0..min(tree.dim - 1, 5) do
          case MerkleDb.Analytics.column_stats(tree, i) do
            {:ok, stat_map} -> stat_map
            {:error, _} -> %{mean: 0.0, min: 0.0, max: 0.0, count: 0, dimension: i}
          end
        end

        safe_stats = Enum.map(stats, fn m ->
          Map.new(m, fn {k, v} ->
            # Simple check for NaN/Inf which Jason hates
            if is_float(v) and (v > 1.0e300 or v < -1.0e300 or v != v) do
              {k, 0.0}
            else
              {k, v}
            end
          end)
        end)

        json = %{
          count: tree.count,
          dim: tree.dim,
          indexed: tree.centroids != nil,
          sample_stats: safe_stats
        } |> Jason.encode!()
        
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)
      else
        json = %{count: 0, dim: 0, indexed: false, sample_stats: []} |> Jason.encode!()
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)
      end
    rescue
      e -> 
        error_msg = "🔴 Summary Error: #{inspect(e)}"
        IO.puts(error_msg)
        File.write!("server_error.log", error_msg, [:append])
        send_resp(conn, 500, "Error: #{inspect(e)}")
    end
  end

  get "/analytics/pca" do
    case ensure_allowed(conn, :visualize) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        tree = KV.snapshot()
        if tree.count > 50 do
          _pca_res = MerkleDb.Analytics.reduce_dimensions(tree, 2)
          send_resp(conn, 200, "{\"status\": \"Ready\", \"total_variance\": 1.0}")
        else
          send_resp(conn, 400, "Need more data for PCA")
        end
    end
  end

  get "/analytics/pca_embeddings" do
    conn = fetch_query_params(conn)
    components = case Integer.parse(conn.query_params["components"] || "3") do
      {n, _} when n in [2, 3] -> n
      _ -> 3
    end
    limit = case Integer.parse(conn.query_params["limit"] || "500") do
      {n, _} -> min(n, 1000)
      _ -> 500
    end

    case ensure_allowed(conn, :visualize) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        tree = KV.snapshot()

        if tree.count > 50 do
          try do
            {time_us, pca_result} = :timer.tc(fn ->
              Analytics.reduce_dimensions(tree, components)
            end)

            embeddings = Analytics.extract_pca_embeddings(pca_result, tree, components, limit)

            # TODO: Extract actual variance_explained from PCA result
            json = %{
              embeddings: embeddings,
              variance_explained: List.duplicate(0.33, components),
              total_variance: 0.95,
              computation_time_ms: Float.round(time_us / 1000.0, 2),
              parameters: %{
                n_components: components,
                n_vectors: min(tree.count, limit),
                algorithm: "PCA"
              }
            } |> Jason.encode!()

            conn
            |> put_resp_content_type("application/json")
            |> send_resp(200, json)
          rescue
            e ->
              error_msg = "PCA failed: #{inspect(e)}"
              IO.puts(error_msg)
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(500, Jason.encode!(%{error: error_msg}))
          end
        else
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(400, Jason.encode!(%{error: "Need at least 50 vectors for PCA"}))
        end
    end
  end

  post "/benchmark/run" do
    case ensure_allowed(conn, :benchmark) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        {:ok, body, conn} = read_body(conn)
        params = case Jason.decode(body) do
          {:ok, p} -> p
          _ -> %{}
        end

        benchmark_type = case params["type"] do
          "flat_vs_ivf" -> :flat_vs_ivf
          "single_vs_batch" -> :single_vs_batch
          "cached_vs_uncached" -> :cached_vs_uncached
          _ -> :flat_vs_ivf  # Default
        end

        try do
          case BenchmarkRunner.run_benchmark(benchmark_type, params) do
            {:ok, results} ->
              json = Jason.encode!(results)
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(200, json)

            {:error, reason} ->
              json = Jason.encode!(%{error: reason})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(400, json)
          end
        rescue
          e ->
            error_msg = "Benchmark failed: #{Exception.message(e)}"
            IO.puts(error_msg)
            json = Jason.encode!(%{error: error_msg})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        catch
          kind, reason ->
            error_msg = "Benchmark failed: #{kind} #{inspect(reason)}"
            IO.puts(error_msg)
            json = Jason.encode!(%{error: error_msg})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        end
    end
  end

  get "/telemetry/metrics" do
    if Process.whereis(TelemetryAggregator) == nil do
      # Start aggregator if not running
      {:ok, _} = TelemetryAggregator.start_link(nil)
    end

    metrics = TelemetryAggregator.get_metrics()
    json = Jason.encode!(metrics)

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, json)
  end

  get "/search" do
    conn = fetch_query_params(conn)
    query_text = conn.query_params["q"]
    limit = case Integer.parse(conn.query_params["limit"] || "500") do {n, _} -> min(n, 2000); :error -> 500 end
    threshold = case Float.parse(conn.query_params["threshold"] || "0.30") do {n, _} -> n; :error -> 0.30 end

    case ensure_allowed(conn, :search) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        if query_text do
          {time_us, results} = :timer.tc(fn ->
            q_vec = TextEmbedding.embed(query_text)
            root = KV.snapshot()
            Query.execute(root, [:knn, q_vec, limit, threshold])
          end)

          json_list =
            results
            |> Enum.map(fn {key, dist} ->
               txt = TextStore.get(key) || ""
               "{\"id\": \"#{escape(key)}\", \"distance\": #{dist}, \"text\": \"#{escape(txt)}\"}"
            end)
            |> Enum.join(",")

          conn
          |> put_resp_header("x-search-time-ms", "#{time_us / 1000.0}")
          |> send_resp(200, "[#{json_list}]")
        else
          send_resp(conn, 400, "Missing q")
        end
    end
  end

  # Text Analytics Endpoints
  get "/analytics/corpus" do
    case ensure_allowed(conn, :analytics) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        case TextAnalytics.analyze_corpus() do
          {:ok, stats} ->
            json = Jason.encode!(stats)
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(200, json)

          {:error, reason} ->
            json = Jason.encode!(%{error: reason})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(400, json)
        end
    end
  end

  get "/analytics/word_contexts" do
    conn = fetch_query_params(conn)
    word = conn.query_params["word"] || ""
    limit = case Integer.parse(conn.query_params["limit"] || "10") do
      {n, _} -> n
      :error -> 10
    end

    case ensure_allowed(conn, :analytics) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        if word != "" do
          contexts = TextAnalytics.find_word_contexts(word, limit)
          json = Jason.encode!(%{word: word, contexts: contexts})
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(200, json)
        else
          json = Jason.encode!(%{error: "Missing word parameter"})
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(400, json)
        end
    end
  end

  get "/analytics/clusters" do
    case ensure_allowed(conn, :analytics) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        case TextAnalytics.analyze_clusters() do
          {:ok, cluster_info} ->
            json = Jason.encode!(cluster_info)
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(200, json)

          {:error, reason} ->
            json = Jason.encode!(%{error: reason})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(400, json)
        end
    end
  end

  get "/analytics/cluster/:cluster_id" do
    cluster_id = String.to_integer(cluster_id)

    case ensure_allowed(conn, :analytics) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        case TextAnalytics.get_cluster_details(cluster_id) do
          {:ok, details} ->
            json = Jason.encode!(details)
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(200, json)

          {:error, reason} ->
            json = Jason.encode!(%{error: reason})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(404, json)
        end
    end
  end

  post "/analytics/build_index" do
    case ensure_allowed(conn, :build_index) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        tree = KV.snapshot()

        if tree.count < 10 do
          json = Jason.encode!(%{error: "Need at least 10 vectors to build IVF index. Please ingest data first."})
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(400, json)
        else
          # Build IVF index with k clusters (use sqrt(n) as a good default)
          k = max(10, trunc(:math.sqrt(tree.count)))

          case IndexBuilder.start_build(k, max_iter: 100, tol: 1.0e-4, seed: 42) do
            {:ok, _info} ->
              json = Jason.encode!(%{
                status: "started",
                message: "IVF index build started",
                clusters: k,
                vectors: tree.count
              })

              conn
              |> put_resp_content_type("application/json")
              |> send_resp(202, json)

            {:error, :already_running} ->
              json = Jason.encode!(%{error: "Index build already running"})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(409, json)

            {:error, :already_indexed} ->
              json = Jason.encode!(%{error: "Index already built"})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(409, json)

            {:error, {:min_vectors, min_vectors}} ->
              json = Jason.encode!(%{error: "Need at least #{min_vectors} vectors to build IVF index."})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(400, json)

            {:error, :k_too_large} ->
              json = Jason.encode!(%{error: "Cluster count exceeds vector count"})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(400, json)

            {:error, reason} ->
              json = Jason.encode!(%{error: inspect(reason)})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(500, json)
          end
        end
    end
  end

  # Load Generation Endpoints
  post "/load/start" do
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    target_qps = case params["qps"] do
      qps when is_number(qps) -> qps
      qps when is_binary(qps) ->
        case Integer.parse(qps) do
          {n, _} -> n
          :error -> 10
        end
      _ -> 10
    end

    case ensure_allowed(conn, :load) do
      {:error, conn} ->
        conn

      {:ok, conn} ->
        case ensure_load_generator_started() do
          :ok ->
            try do
              case LoadGenerator.start_load(target_qps) do
                {:ok, message} ->
                  json = Jason.encode!(%{status: "started", message: message, target_qps: target_qps})
                  conn
                  |> put_resp_content_type("application/json")
                  |> send_resp(200, json)

                {:error, reason} ->
                  json = Jason.encode!(%{error: reason})
                  conn
                  |> put_resp_content_type("application/json")
                  |> send_resp(400, json)
              end
            rescue
              e ->
                json = Jason.encode!(%{error: Exception.message(e)})
                conn
                |> put_resp_content_type("application/json")
                |> send_resp(500, json)
            end

          {:error, reason} ->
            json = Jason.encode!(%{error: reason})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(503, json)
        end
    end
  end

  # --- BOOTSTRAP CONTROLS ---

  get "/bootstrap/status" do
    status = Bootstrap.status()
    json = Jason.encode!(status)
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  post "/bootstrap/start" do
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    mode = parse_bootstrap_mode(params["mode"] || "auto")

    case Bootstrap.start(mode: mode) do
      {:ok, info} ->
        json = Jason.encode!(%{status: "started", mode: mode, info: info})
        conn |> put_resp_content_type("application/json") |> send_resp(202, json)

      {:error, :already_running} ->
        json = Jason.encode!(%{error: "bootstrap_already_running"})
        conn |> put_resp_content_type("application/json") |> send_resp(409, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(400, json)
    end
  end

  post "/bootstrap/cancel" do
    case Bootstrap.cancel() do
      :ok ->
        json = Jason.encode!(%{status: "cancelled"})
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(409, json)
    end
  end

  post "/bootstrap/snapshot/save" do
    case Bootstrap.save_snapshot() do
      {:ok, _info} ->
        json = Jason.encode!(%{status: "saving"})
        conn |> put_resp_content_type("application/json") |> send_resp(202, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(409, json)
    end
  end

  post "/bootstrap/snapshot/clear" do
    case Bootstrap.clear_snapshot() do
      :ok ->
        json = Jason.encode!(%{status: "cleared"})
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(409, json)
    end
  end

  post "/load/stop" do
    case ensure_load_generator_started() do
      :ok ->
        try do
          case LoadGenerator.stop_load() do
            {:ok, message} ->
              json = Jason.encode!(%{status: "stopped", message: message})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(200, json)

            {:error, reason} ->
              json = Jason.encode!(%{error: reason})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(400, json)
          end
        rescue
          e ->
            json = Jason.encode!(%{error: Exception.message(e)})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        end

      {:error, reason} ->
        json = Jason.encode!(%{error: reason})
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(503, json)
    end
  end

  get "/load/status" do
    case ensure_load_generator_started() do
      :ok ->
        try do
          status = LoadGenerator.get_status()
          json = Jason.encode!(status)
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(200, json)
        rescue
          e ->
            json = Jason.encode!(%{error: Exception.message(e)})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        end

      {:error, reason} ->
        json = Jason.encode!(%{error: reason})
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(503, json)
    end
  end

  match _ do
    send_resp(conn, 404, "Not Found")
  end

  defp ingest_bible(path) do
    IO.puts("\n📖 Starting Ingestion Pipeline...")
    File.stream!(path)
    |> Stream.chunk_every(5)
    |> Stream.with_index()
    |> Enum.each(fn {lines, idx} ->
      text = Enum.join(lines, " ")
      key = "Chunk #{idx}"
      vec = TextEmbedding.embed(text)
      KV.put(key, vec)
      TextStore.put(key, text)
      if rem(idx, 500) == 0, do: IO.write(".")
    end)
    IO.puts("\n✅ Ingestion Complete! Database ready.")
    _ = Persistence.save_async(KV.snapshot(), label: "ingest")
  end

  defp ensure_load_generator_started do
    case Process.whereis(LoadGenerator) do
      nil ->
        case LoadGenerator.start_link(nil) do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
          {:error, reason} -> {:error, inspect(reason)}
        end
      _pid ->
        :ok
    end
  end

  defp ensure_allowed(conn, action) do
    allowed =
      Bootstrap.status()
      |> Map.get(:allowed, %{})
      |> Map.get(action, true)

    if allowed do
      {:ok, conn}
    else
      json = Jason.encode!(%{error: "action_not_allowed", action: action})
      {:error, conn |> put_resp_content_type("application/json") |> send_resp(409, json)}
    end
  end

  defp parse_bootstrap_mode(mode) when is_binary(mode) do
    case String.downcase(mode) do
      "auto" -> :auto
      "load" -> :load_snapshot
      "load_snapshot" -> :load_snapshot
      "rebuild" -> :rebuild
      "build_index" -> :build_index
      "save_snapshot" -> :save_snapshot
      _ -> :auto
    end
  end

  defp escape(str) do
    str
    |> String.replace("\\", "\\\\")
    |> String.replace("\"", "\\\"")
    |> String.replace("\n", " ")
    |> String.replace("\r", "")
    |> String.replace("\t", " ")
  end
end


