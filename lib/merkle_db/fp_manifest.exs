defmodule MerkleDb.FP.Manifest do
  @moduledoc """
  Manifest and auto-dispatch rules for FP_ASM_LIB functions.

  This file is evaluated by generators to drive:
  - full API coverage
  - automatic sync/async/parallel dispatch
  - per-function overrides
  """

  @type mode :: :auto | :sync | :async | :parallel
  @type cost_class :: :fast | :medium | :long

  @class_defaults %{
    fast: %{
      cost_class: :fast,
      async_min_bytes: 8_000_000,
      parallel_min_bytes: 16_000_000,
      supports_async: false,
      supports_parallel: true,
      parallel_strategy: :chunked_elixir
    },
    medium: %{
      cost_class: :medium,
      async_min_bytes: 2_000_000,
      parallel_min_bytes: 8_000_000,
      supports_async: true,
      supports_parallel: true,
      parallel_strategy: :chunked_elixir
    },
    long: %{
      cost_class: :long,
      async_min_bytes: 0,
      parallel_min_bytes: 0,
      supports_async: true,
      supports_parallel: false,
      parallel_strategy: :none
    }
  }

  @manifest %{
    version: 1,
    defaults: %{
      mode: :auto,
      cost_class: :fast,
      supports_async: false,
      supports_parallel: false,
      parallel_strategy: :none
    },
    class_defaults: @class_defaults,
    rules: [
      %{match: ~r/^fp_kmeans_/, cost_class: :long, progress: :iter},
      %{match: ~r/^fp_pca_/, cost_class: :long, progress: :iter},
      %{match: ~r/^fp_neural_network_/, cost_class: :long, progress: :epoch},
      %{match: ~r/^fp_gaussian_nb_train$/, cost_class: :long, progress: :iter},
      %{match: ~r/^fp_multinomial_nb_train$/, cost_class: :long, progress: :iter},
      %{match: ~r/^fp_(rolling|percentile|moments|detect_outliers|sma|wma|ema)_/, cost_class: :medium},
      %{match: ~r/^fp_(map|zip|scan|reduce|fold|range|replicate|reverse|slice|take|drop|filter|partition|group|unique|union|intersect|pred)_/, cost_class: :fast}
    ],
    overrides: %{
      "fp_linear_regression_r2_score" => %{cost_class: :fast, supports_parallel: false},
      "fp_gaussian_nb_predict_batch" => %{cost_class: :medium, supports_parallel: true},
      "fp_multinomial_nb_predict_batch" => %{cost_class: :medium, supports_parallel: true},
      "fp_pca_generate_low_rank_data" => %{cost_class: :long},
      "fp_pca_generate_ellipse_data" => %{cost_class: :medium}
    }
  }

  def manifest, do: @manifest

  def resolve_meta(func_name) do
    base =
      @manifest.defaults
      |> Map.merge(class_defaults(@manifest.defaults.cost_class))

    base
    |> Map.merge(match_rule(func_name))
    |> Map.merge(Map.get(@manifest.overrides, func_name, %{}))
  end

  def dispatch(func_name, metrics, opts \\ []) do
    mode = Keyword.get(opts, :mode, @manifest.defaults.mode)
    if mode != :auto, do: mode, else: auto_dispatch(func_name, metrics)
  end

  defp auto_dispatch(func_name, metrics) do
    meta = resolve_meta(func_name)
    bytes_in = Map.get(metrics, :bytes_in, 0)
    bytes_out = Map.get(metrics, :bytes_out, 0)
    total_bytes = bytes_in + bytes_out

    cond do
      meta.supports_async and total_bytes >= meta.async_min_bytes -> :async
      meta.supports_parallel and total_bytes >= meta.parallel_min_bytes -> :parallel
      true -> :sync
    end
  end

  defp class_defaults(cost_class) do
    Map.get(@class_defaults, cost_class, @class_defaults.fast)
  end

  defp match_rule(func_name) do
    Enum.find_value(@manifest.rules, %{}, fn rule ->
      if Regex.match?(rule.match, func_name), do: Map.delete(rule, :match)
    end)
  end
end
