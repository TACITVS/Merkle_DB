defmodule MerkleDb.Web.Router do
  @moduledoc """
  HTTP router for MerkleDb API.

  ## Security
  - All /v1 endpoints require authentication
  - Rate limiting applied globally
  - Body size limits enforced
  - Input validation on all write operations

  ## Health Checks
  - GET /health/live - Liveness probe (is the process running?)
  - GET /health/ready - Readiness probe (can serve traffic?)
  """
  use Plug.Router
  require Logger

  alias MerkleDb.{Bootstrap, Filter, KV, PayloadStore, Persistence, Query, Replication,
                  TextEmbedding, TextStore, JobScheduler, Analytics, BenchmarkRunner,
                  TelemetryAggregator, TextAnalytics, LoadGenerator, IndexBuilder,
                  FPDispatcher, Validator, RateLimiter}
  alias MerkleDb.Web.Auth

  @max_body_size Application.compile_env(:merkle_db, :max_request_body_bytes, 10_485_760)

  # Static assets (no auth required)
  plug Plug.Static,
    at: "/",
    from: {:merkle_db, "priv/static"},
    only: ["index.html"]

  plug :match
  plug :rate_limit_ip
  plug :authenticate
  plug Plug.Parsers,
    parsers: [:json],
    pass: ["application/json"],
    json_decoder: Jason,
    body_reader: {__MODULE__, :read_body_with_limit, []}
  plug :dispatch

  # Custom body reader with size limit
  def read_body_with_limit(conn, opts) do
    max_size = Keyword.get(opts, :length, @max_body_size)
    Plug.Conn.read_body(conn, length: max_size)
  end

  # ============================================================================
  # MIDDLEWARE
  # ============================================================================

  # Rate limiting by IP (applied before authentication)
  defp rate_limit_ip(conn, _opts) do
    # Skip rate limiting for health checks
    if health_check_path?(conn.request_path) do
      conn
    else
      case RateLimiter.check_rate_ip(conn) do
        :ok ->
          conn

        {:error, :rate_limited, retry_after} ->
          conn
          |> put_resp_header("retry-after", to_string(div(retry_after, 1000)))
          |> put_resp_content_type("application/json")
          |> send_resp(429, Jason.encode!(%{error: "rate_limited", retry_after_ms: retry_after}))
          |> halt()
      end
    end
  end

  # Authentication middleware
  defp authenticate(conn, _opts) do
    cond do
      # Health checks - no auth required
      health_check_path?(conn.request_path) ->
        conn

      # API V1 - auth required with scope based on method
      String.starts_with?(conn.request_path, "/v1") ->
        scope = scope_for_method(conn.method)
        Auth.call(conn, scope: scope)

      # Admin endpoints - require admin scope
      String.starts_with?(conn.request_path, "/admin") ->
        Auth.call(conn, scope: :admin)

      # Other endpoints - check if auth is globally required
      Application.get_env(:merkle_db, :require_auth, false) ->
        Auth.call(conn, [])

      # No auth required in dev/test mode
      true ->
        conn
    end
  end

  defp health_check_path?("/health" <> _), do: true
  defp health_check_path?("/favicon.ico"), do: true
  defp health_check_path?(_), do: false

  defp scope_for_method("GET"), do: :read
  defp scope_for_method("HEAD"), do: :read
  defp scope_for_method("OPTIONS"), do: :read
  defp scope_for_method(_), do: :write

  # ============================================================================
  # HEALTH CHECKS (No auth required)
  # ============================================================================

  @doc """
  Liveness probe - is the process running?
  Returns 200 if the Erlang VM is up.
  """
  get "/health/live" do
    json = Jason.encode!(%{
      status: "ok",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    })
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  @doc """
  Readiness probe - can the service accept traffic?
  Checks that essential services are running.
  """
  get "/health/ready" do
    checks = %{
      kv_store: check_kv_store(),
      raft: check_raft(),
      rate_limiter: check_rate_limiter(),
      api_key_store: check_api_key_store()
    }

    all_healthy = Enum.all?(checks, fn {_, status} -> status == :ok end)

    response = %{
      status: if(all_healthy, do: "ready", else: "degraded"),
      checks: Map.new(checks, fn {k, v} -> {k, to_string(v)} end),
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    status_code = if all_healthy, do: 200, else: 503
    json = Jason.encode!(response)
    conn |> put_resp_content_type("application/json") |> send_resp(status_code, json)
  end

  @doc """
  Detailed health check with metrics.
  """
  get "/health/detailed" do
    tree = KV.snapshot()

    response = %{
      status: "ok",
      version: Application.spec(:merkle_db, :vsn) |> to_string(),
      uptime_seconds: :erlang.statistics(:wall_clock) |> elem(0) |> div(1000),
      memory: %{
        total_mb: :erlang.memory(:total) |> div(1_048_576),
        processes_mb: :erlang.memory(:processes) |> div(1_048_576),
        ets_mb: :erlang.memory(:ets) |> div(1_048_576)
      },
      database: %{
        vector_count: tree.count,
        dimensions: tree.dim,
        indexed: tree.centroids != nil,
        collections: length(KV.list_collections())
      },
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    json = Jason.encode!(response)
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  # ============================================================================
  # API V1 (Authenticated)
  # ============================================================================

  get "/v1/collections" do
    collections = KV.list_collections()
    json = Jason.encode!(%{collections: collections})
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  post "/v1/collections/:collection" do
    with :ok <- Validator.validate_collection_name(collection),
         {:ok, body, conn} <- read_body(conn),
         {:ok, params} <- parse_json(body) do
      opts =
        params
        |> Enum.map(fn {k, v} -> {String.to_atom(k), v} end)
        |> Enum.map(fn
          {:precision, p} when is_binary(p) -> {:precision, String.to_atom(p)}
          other -> other
        end)

      case KV.create_collection(collection, opts) do
        :ok ->
          Logger.info("Collection '#{collection}' created")
          send_json(conn, 201, %{status: "ok", message: "Collection created"})

        {:error, :already_exists} ->
          send_json(conn, 409, %{error: "already_exists"})

        err ->
          send_json(conn, 500, %{error: inspect(err)})
      end
    else
      {:error, reason} when is_binary(reason) ->
        send_json(conn, 400, %{error: reason})

      {:error, :invalid_json} ->
        send_json(conn, 400, %{error: "Invalid JSON body"})

      err ->
        send_json(conn, 400, %{error: inspect(err)})
    end
  end

  delete "/v1/collections/:collection" do
    with :ok <- Validator.validate_collection_name(collection) do
      case KV.drop_collection(collection) do
        :ok ->
          Logger.info("Collection '#{collection}' dropped")
          send_json(conn, 200, %{status: "ok", message: "Collection dropped"})

        err ->
          send_json(conn, 500, %{error: inspect(err)})
      end
    else
      {:error, reason} ->
        send_json(conn, 400, %{error: reason})
    end
  end

  post "/v1/:collection/checkpoint" do
    with :ok <- Validator.validate_collection_name(collection) do
      case KV.checkpoint(collection) do
        :ok ->
          send_json(conn, 200, %{status: "ok", message: "Checkpoint created"})

        {:error, reason} ->
          send_json(conn, 500, %{error: inspect(reason)})
      end
    else
      {:error, reason} ->
        send_json(conn, 400, %{error: reason})
    end
  end

  post "/v1/:collection/vectors" do
    with :ok <- Validator.validate_collection_name(collection),
         {:ok, body, conn} <- read_body(conn),
         {:ok, items} when is_list(items) <- parse_json(body),
         :ok <- Validator.validate_batch(items) do

      batch = Enum.map(items, fn item ->
        id = item["id"]
        meta = item["metadata"] || %{}

        vec =
          cond do
            item["vector"] ->
              for f <- item["vector"], into: <<>>, do: <<f::float-little-32>>

            item["text"] ->
              TextEmbedding.embed(item["text"])

            true -> <<>>
          end

        {id, vec, meta}
      end)

      case KV.put_batch(collection, batch) do
        :ok ->
          send_json(conn, 200, %{status: "ok", count: length(batch)})

        {:error, reason} ->
          send_json(conn, 500, %{error: inspect(reason)})
      end
    else
      {:error, reason} when is_binary(reason) ->
        send_json(conn, 400, %{error: reason})

      {:ok, _} ->
        send_json(conn, 400, %{error: "Expected array of vectors"})

      err ->
        send_json(conn, 400, %{error: inspect(err)})
    end
  end

  post "/v1/:collection/search" do
    with :ok <- Validator.validate_collection_name(collection),
         {:ok, body, conn} <- read_body(conn),
         {:ok, params} <- parse_json(body),
         :ok <- Validator.validate_query_params(params) do

      k = params["k"] || 10
      threshold = params["threshold"] || 0.0

      query =
        cond do
          params["vector"] ->
            vec_bin = for f <- params["vector"], into: <<>>, do: <<f::float-little-32>>
            [:knn, vec_bin, k, threshold]

          params["text"] ->
            [:semantic, params["text"], k, threshold]

          true -> nil
        end

      if query do
        tree = KV.snapshot(collection)
        results = Query.execute(tree, query)

        hits = Enum.map(results, fn {key, score} ->
          %{id: key, score: score}
        end)

        send_json(conn, 200, %{results: hits})
      else
        send_json(conn, 400, %{error: "Missing 'vector' or 'text' in body"})
      end
    else
      {:error, reason} when is_binary(reason) ->
        send_json(conn, 400, %{error: reason})

      err ->
        send_json(conn, 400, %{error: inspect(err)})
    end
  end

  # ============================================================================
  # JOB CONTROLS
  # ============================================================================

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

  # ============================================================================
  # STANDARD ENDPOINTS
  # ============================================================================

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
          path = Application.get_env(:merkle_db, :ingest_file_path, "C:/Users/baian/AppData/Roaming/nltk_data/corpora/gutenberg/bible-kjv.txt")
          if File.exists?(path) do
            LoadGenerator.stop_if_active()
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
        stats = for i <- 0..min(tree.dim - 1, 5) do
          case MerkleDb.Analytics.column_stats(tree, i) do
            {:ok, stat_map} -> stat_map
            {:error, _} -> %{mean: 0.0, min: 0.0, max: 0.0, count: 0, dimension: i}
          end
        end

        safe_stats = Enum.map(stats, fn m ->
          Map.new(m, fn {k, v} ->
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
        Logger.error("Summary Error: #{inspect(e)}")
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
          pca_res = Analytics.reduce_dimensions(tree, 2)
          stats = Analytics.pca_stats(pca_res)

          json =
            %{
              status: "Ready",
              total_variance: stats.total_variance,
              variance_explained: stats.explained_variance
            }
            |> Jason.encode!()

          conn
          |> put_resp_content_type("application/json")
          |> send_resp(200, json)
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
            stats = Analytics.pca_stats(pca_result)

            json =
              %{
                embeddings: embeddings,
                variance_explained: Enum.take(stats.explained_variance, components),
                total_variance: stats.total_variance,
                computation_time_ms: Float.round(time_us / 1000.0, 2),
                parameters: %{
                  n_components: components,
                  n_vectors: min(tree.count, limit),
                  algorithm: "PCA"
                }
              }
              |> Jason.encode!()

            conn
            |> put_resp_content_type("application/json")
            |> send_resp(200, json)
          rescue
            e ->
              error_msg = "PCA failed: #{inspect(e)}"
              Logger.error(error_msg)
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
          _ -> :flat_vs_ivf
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
            Logger.error(error_msg)
            json = Jason.encode!(%{error: error_msg})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        catch
          kind, reason ->
            error_msg = "Benchmark failed: #{kind} #{inspect(reason)}"
            Logger.error(error_msg)
            json = Jason.encode!(%{error: error_msg})
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(500, json)
        end
    end
  end

  get "/telemetry/metrics" do
    if Process.whereis(TelemetryAggregator) == nil do
      {:ok, _} = TelemetryAggregator.start_link(nil)
    end

    metrics = TelemetryAggregator.get_metrics()
    json = Jason.encode!(metrics)

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, json)
  end

  get "/fp/dispatcher/status" do
    case ensure_fp_dispatcher_started() do
      :ok ->
        status = FPDispatcher.status()
        json = Jason.encode!(status)
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: reason})
        conn |> put_resp_content_type("application/json") |> send_resp(503, json)
    end
  end

  get "/fp/jobs/result" do
    conn = fetch_query_params(conn)

    case ensure_fp_dispatcher_started() do
      :ok ->
        case parse_job_id(conn.query_params["job_id"]) do
          {:ok, job_id} ->
            case FPDispatcher.result(job_id) do
              {:ok, result} ->
                json = Jason.encode!(%{status: "ok", job_id: job_id, result: normalize_fp_result(result)})
                conn |> put_resp_content_type("application/json") |> send_resp(200, json)

              {:error, :queued} ->
                json = Jason.encode!(%{status: "queued", job_id: job_id})
                conn |> put_resp_content_type("application/json") |> send_resp(200, json)

              {:error, :running} ->
                json = Jason.encode!(%{status: "running", job_id: job_id})
                conn |> put_resp_content_type("application/json") |> send_resp(200, json)

              {:error, :unknown_job} ->
                send_fp_error(conn, 404, "unknown_job")

              {:error, reason} ->
                send_fp_error(conn, 400, reason)
            end

          :error ->
            send_fp_error(conn, 400, "invalid_job_id")
        end

      {:error, reason} ->
        send_fp_error(conn, 503, reason)
    end
  end

  post "/fp/jobs/submit" do
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    name = params["name"]
    args = params["args"]
    mode = params["mode"]

    case ensure_fp_dispatcher_started() do
      :ok ->
        cond do
          not is_binary(name) or name == "" ->
            send_fp_error(conn, 400, "missing_function_name")

          not is_list(args) ->
            send_fp_error(conn, 400, "args_must_be_array")

          true ->
            opts =
              case mode do
                nil -> []
                "auto" -> []
                mode when is_binary(mode) -> [mode: mode]
                _ -> []
              end

            try do
              case FPDispatcher.submit(name, args, opts) do
                {:ok, job_id} ->
                  json = Jason.encode!(%{status: "queued", job_id: job_id})
                  conn |> put_resp_content_type("application/json") |> send_resp(202, json)

                {:error, reason} ->
                  send_fp_error(conn, 400, reason)
              end
            rescue
              e ->
                send_fp_error(conn, 400, Exception.message(e))
            end
        end

      {:error, reason} ->
        send_fp_error(conn, 503, reason)
    end
  end

  post "/fp/jobs/cancel" do
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    case parse_job_id(params["job_id"]) do
      {:ok, job_id} ->
        case FPDispatcher.cancel(job_id) do
          {:ok, stage} ->
            json = Jason.encode!(%{status: "canceled", stage: stage, job_id: job_id})
            conn |> put_resp_content_type("application/json") |> send_resp(200, json)

          {:error, :already_finished} ->
            send_fp_error(conn, 409, "already_finished")

          {:error, :unknown_job} ->
            send_fp_error(conn, 404, "unknown_job")

          {:error, reason} ->
            send_fp_error(conn, 400, reason)
        end

      :error ->
        send_fp_error(conn, 400, "invalid_job_id")
    end
  end

  post "/fp/jobs/stop" do
    case ensure_fp_dispatcher_started() do
      :ok ->
        case FPDispatcher.cancel_all() do
          %{canceled_queue: queued, canceled_running: running} ->
            json =
              Jason.encode!(%{
                status: "stopped",
                canceled_queue: queued,
                canceled_running: running,
                message: "Canceled #{queued} queued and #{running} running jobs"
              })

            conn |> put_resp_content_type("application/json") |> send_resp(200, json)

          {:error, reason} ->
            send_fp_error(conn, 400, reason)
        end

      {:error, reason} ->
        send_fp_error(conn, 503, reason)
    end
  end

  post "/fp/jobs/flush" do
    case ensure_fp_dispatcher_started() do
      :ok ->
        case FPDispatcher.flush_queue() do
          flushed when is_integer(flushed) ->
            json = Jason.encode!(%{status: "flushed", flushed: flushed})
            conn |> put_resp_content_type("application/json") |> send_resp(200, json)

          {:error, reason} ->
            send_fp_error(conn, 400, reason)
        end

      {:error, reason} ->
        send_fp_error(conn, 503, reason)
    end
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
          filter_param = conn.query_params["filter"]
          filter_result = Filter.parse_query_param(filter_param)

          case filter_result do
            {:error, reason} ->
              json = Jason.encode!(%{error: "Invalid filter: #{reason}"})
              conn
              |> put_resp_content_type("application/json")
              |> send_resp(400, json)

            {:ok, filter} ->
              tree = KV.snapshot()
              is_indexed = tree.centroids != nil

              {time_us, results} = :timer.tc(fn ->
                q_vec = TextEmbedding.embed(query_text)

                if filter == [] do
                  Query.execute(tree, [:knn, q_vec, limit, threshold])
                else
                  Query.execute(tree, [:knn, q_vec, limit, threshold, {:where, filter}])
                end
              end)

              hits =
                results
                |> Enum.map(fn {key, dist} ->
                  txt = TextStore.get(key)
                  payload = PayloadStore.get(key)
                  %{
                    id: key,
                    distance: dist,
                    text: if(txt in [nil, ""], do: "Text not found", else: txt),
                    payload: payload
                  }
                end)

              response = %{
                results: hits,
                indexed: is_indexed,
                filtered: filter != [],
                count: length(hits),
                search_time_ms: time_us / 1000.0
              }

              conn
              |> put_resp_header("x-search-time-ms", "#{time_us / 1000.0}")
              |> put_resp_content_type("application/json")
              |> send_resp(200, Jason.encode!(response))
          end
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
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    force = params["force"] == true

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
          k = params["k"] || max(10, trunc(:math.sqrt(tree.count)))

          case IndexBuilder.start_build(k, max_iter: 100, tol: 1.0e-4, seed: 42, force: force) do
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
              json = Jason.encode!(%{error: "Index already built. Use {\"force\": true} to rebuild."})
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

  # ============================================================================
  # BOOTSTRAP CONTROLS
  # ============================================================================

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

  # ============================================================================
  # REPLICATION ENDPOINTS
  # ============================================================================

  get "/replication/status" do
    status = Replication.status()
    json = Jason.encode!(status)
    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  get "/replication/deltas" do
    conn = fetch_query_params(conn)
    since = case Integer.parse(conn.query_params["since"] || "0") do
      {n, _} -> n
      :error -> 0
    end
    limit = case Integer.parse(conn.query_params["limit"] || "1000") do
      {n, _} -> min(n, 10000)
      :error -> 1000
    end

    {:ok, ops} = Replication.get_deltas(since: since, limit: limit)

    json_ops = Enum.map(ops, fn op ->
      %{
        seq: op.seq,
        op: op.op,
        key: op.key,
        data: serialize_operation_data(op.data),
        timestamp: op.timestamp
      }
    end)

    json = Jason.encode!(%{
      current_seq: Replication.current_seq(),
      operations: json_ops,
      count: length(json_ops)
    })

    conn |> put_resp_content_type("application/json") |> send_resp(200, json)
  end

  post "/replication/apply" do
    {:ok, body, conn} = read_body(conn)

    case Jason.decode(body) do
      {:ok, %{"operations" => operations}} when is_list(operations) ->
        case Replication.apply_operations(operations) do
          {:ok, count} ->
            json = Jason.encode!(%{status: "applied", count: count})
            conn |> put_resp_content_type("application/json") |> send_resp(200, json)

          {:error, reason} ->
            json = Jason.encode!(%{error: inspect(reason)})
            conn |> put_resp_content_type("application/json") |> send_resp(400, json)
        end

      {:ok, _} ->
        json = Jason.encode!(%{error: "Missing operations array"})
        conn |> put_resp_content_type("application/json") |> send_resp(400, json)

      {:error, _} ->
        json = Jason.encode!(%{error: "Invalid JSON"})
        conn |> put_resp_content_type("application/json") |> send_resp(400, json)
    end
  end

  get "/replication/snapshot" do
    case Replication.export_snapshot() do
      {:ok, snapshot} ->
        json_snapshot = %{
          type: snapshot.type,
          timestamp: snapshot.timestamp,
          seq: snapshot.seq,
          tree_stats: snapshot.tree_stats,
          vector_count: length(snapshot.vectors || []),
          payload_count: safe_map_size(snapshot.payloads),
          text_count: safe_map_size(snapshot.texts)
        }

        json = Jason.encode!(json_snapshot)
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(500, json)
    end
  end

  post "/replication/compact" do
    {:ok, body, conn} = read_body(conn)
    params = case Jason.decode(body) do
      {:ok, p} -> p
      _ -> %{}
    end

    keep_last = params["keep_last"] || 10000

    case Replication.compact(keep_last: keep_last) do
      {:ok, deleted} ->
        json = Jason.encode!(%{status: "compacted", deleted: deleted, keep_last: keep_last})
        conn |> put_resp_content_type("application/json") |> send_resp(200, json)

      {:error, reason} ->
        json = Jason.encode!(%{error: inspect(reason)})
        conn |> put_resp_content_type("application/json") |> send_resp(500, json)
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

  # ============================================================================
  # PRIVATE HELPERS
  # ============================================================================

  defp send_json(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(data))
  end

  defp parse_json(body) when is_binary(body) do
    case Jason.decode(body) do
      {:ok, data} -> {:ok, data}
      {:error, _} -> {:error, :invalid_json}
    end
  end
  defp parse_json(_), do: {:error, :invalid_json}

  defp check_kv_store do
    if Process.whereis(MerkleDb.KV), do: :ok, else: :error
  end

  defp check_raft do
    if Process.whereis(MerkleDb.Raft), do: :ok, else: :not_running
  end

  defp check_rate_limiter do
    if Process.whereis(MerkleDb.RateLimiter), do: :ok, else: :not_running
  end

  defp check_api_key_store do
    if Process.whereis(MerkleDb.ApiKeyStore), do: :ok, else: :not_running
  end

  defp ingest_bible(path) do
    Logger.info("Starting Ingestion Pipeline...")
    File.stream!(path)
    |> Stream.chunk_every(5)
    |> Stream.with_index()
    |> Enum.each(fn {lines, idx} ->
      text = Enum.join(lines, " ")
      key = "Chunk #{idx}"
      vec = TextEmbedding.embed(text)
      KV.put(key, vec)
      TextStore.put(key, text)
      if rem(idx, 500) == 0, do: Logger.debug("Ingested #{idx} chunks")
    end)
    Logger.info("Ingestion Complete! Database ready.")
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

  defp ensure_fp_dispatcher_started do
    case Process.whereis(FPDispatcher) do
      nil ->
        case FPDispatcher.start_link(nil) do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
          {:error, reason} -> {:error, inspect(reason)}
        end

      _pid ->
        :ok
    end
  end

  defp send_fp_error(conn, status, reason) do
    json = Jason.encode!(%{error: normalize_error(reason)})
    conn |> put_resp_content_type("application/json") |> send_resp(status, json)
  end

  defp normalize_error(reason) when is_binary(reason), do: reason
  defp normalize_error(reason), do: inspect(reason)

  defp parse_job_id(job_id) when is_integer(job_id), do: {:ok, job_id}

  defp parse_job_id(job_id) when is_binary(job_id) do
    case Integer.parse(job_id) do
      {value, _} -> {:ok, value}
      :error -> :error
    end
  end

  defp parse_job_id(_), do: :error

  defp normalize_fp_result(result) do
    normalize_fp_value(result)
  end

  defp normalize_fp_value(value) when is_binary(value) do
    %{
      type: "binary",
      size: byte_size(value),
      encoding: "base64",
      value: Base.encode64(value)
    }
  end

  defp normalize_fp_value(value) when is_integer(value) or is_float(value) or is_boolean(value), do: value
  defp normalize_fp_value(value) when is_atom(value), do: Atom.to_string(value)

  defp normalize_fp_value(value) when is_list(value) do
    Enum.map(value, &normalize_fp_value/1)
  end

  defp normalize_fp_value(value) when is_map(value) do
    Map.new(value, fn {k, v} -> {normalize_fp_key(k), normalize_fp_value(v)} end)
  end

  defp normalize_fp_value(value) when is_tuple(value) do
    %{
      type: "tuple",
      value: value |> Tuple.to_list() |> Enum.map(&normalize_fp_value/1)
    }
  end

  defp normalize_fp_value(value) do
    %{type: "term", value: inspect(value)}
  end

  defp normalize_fp_key(key) when is_binary(key), do: key
  defp normalize_fp_key(key) when is_atom(key), do: Atom.to_string(key)
  defp normalize_fp_key(key) when is_integer(key), do: Integer.to_string(key)
  defp normalize_fp_key(key), do: inspect(key)

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

  defp serialize_operation_data(nil), do: nil
  defp serialize_operation_data(data) when is_map(data) do
    data
    |> Map.new(fn
      {:vector, vec} when is_binary(vec) ->
        {"vector", Base.encode64(vec)}
      {key, val} when is_atom(key) ->
        {Atom.to_string(key), serialize_value(val)}
      {key, val} ->
        {key, serialize_value(val)}
    end)
  end

  defp serialize_value(val) when is_binary(val) and byte_size(val) > 100 do
    Base.encode64(val)
  end
  defp serialize_value(val), do: val

  defp safe_map_size(nil), do: 0
  defp safe_map_size(list) when is_list(list), do: length(list)
  defp safe_map_size(map) when is_map(map), do: map_size(map)
  defp safe_map_size(_), do: 0
end
