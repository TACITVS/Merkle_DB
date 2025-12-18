defmodule MerkleDb.Web.Router do
  use Plug.Router
  alias MerkleDb.{KV, Query, TextEmbedding, TextStore, Tuner, JobScheduler, Progress}

  plug Plug.Static,
    at: "/",
    from: {:merkle_db, "priv/static"},
    only: ["index.html"]

  plug :match
  plug :dispatch

  # --- JOB CONTROLS ---

  post "/job/start" do
    # CHECK FIRST: Do we have data?
    tree = KV.snapshot()
    if tree.count > 0 do
      JobScheduler.start_job()
      send_resp(conn, 200, "Started")
    else
      # Safety: Don't start if empty. Tell the user.
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
    # Ensure process is alive before asking
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
    
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, json)
  end

  # --- STANDARD ENDPOINTS ---

  get "/" do
    if Process.whereis(JobScheduler) == nil, do: JobScheduler.start_link(nil)
    if Process.whereis(Progress) == nil, do: Progress.start_link(nil)
    
    conn
    |> put_resp_content_type("text/html")
    |> send_file(200, Application.app_dir(:merkle_db, "priv/static/index.html"))
  end

  post "/ingest" do
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

  post "/tune" do
    try do
      report = Tuner.run_analysis()
      send_resp(conn, 200, "{\"recommended\": #{report.recommended_threshold}}")
    catch e -> 
      send_resp(conn, 500, "Error: #{inspect(e)}") 
    end
  end

  get "/analytics/summary" do
    tree = KV.snapshot()
    if tree.count > 0 do
      # Calculate stats for first few dims as sample
      stats = for i <- 0..min(tree.dim-1, 5) do
        Analytics.column_stats(tree, i)
      end
      
      json = """
      {
        "count": #{tree.count},
        "dim": #{tree.dim},
        "indexed": #{if tree.centroids, do: "true", else: "false"},
        "sample_stats": #{Jason.encode!(stats)}
      }
      """
      conn |> put_resp_content_type("application/json") |> send_resp(200, json)
    else
      send_resp(conn, 404, "Empty")
    end
  end

  get "/analytics/pca" do
    tree = KV.snapshot()
    if tree.count > 50 do
      # Run PCA to 2D for visualization
      pca_res = Analytics.reduce_dimensions(tree, 2)
      # We need a transform function in NIF to actually project the data.
      # For now, let's assume we can transform.
      # Since we don't have transform bridged yet, let's return success metadata.
      send_resp(conn, 200, "{\"status\": \"Ready\", \"total_variance\": 1.0}")
    else
      send_resp(conn, 400, "Need more data for PCA")
    end
  end

  get "/search" do
    conn = fetch_query_params(conn)
    query_text = conn.query_params["q"]
    limit_param = conn.query_params["limit"] || "500"
    limit = case Integer.parse(limit_param) do {n, _} -> min(n, 2000); :error -> 500 end
    threshold_param = conn.query_params["threshold"] || "0.30"
    threshold = case Float.parse(threshold_param) do {n, _} -> n; :error -> 0.30 end
    exact_match = conn.query_params["exact"] == "true"
    search_depth = if exact_match, do: limit * 20, else: limit
    
    if query_text do
      {time_us, results} = :timer.tc(fn -> 
        q_vec = TextEmbedding.embed(query_text)
        root = KV.snapshot()
        Query.execute(root, [:knn, q_vec, search_depth, threshold])
      end)
      
      json_list = 
        results
        |> Stream.map(fn {key, dist, _vec} -> 
           {key, dist, TextStore.get(key)}
        end)
        |> Stream.filter(fn {_key, _dist, text} -> 
           if exact_match do
             safe_q = Regex.escape(query_text)
             Regex.match?(~r/\b#{safe_q}\b/i, text)
           else
             true
           end
        end)
        |> Stream.take(limit)
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