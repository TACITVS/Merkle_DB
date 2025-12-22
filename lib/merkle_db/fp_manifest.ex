defmodule MerkleDb.FPManifest do
  @moduledoc false

  @source_module MerkleDb.ASM
  @entries_cache_key {__MODULE__, :entries}
  @report_cache_key {__MODULE__, :report}
  @default_dir "docs"

  @include_prefixes ["fp_", "get_"]
  @exclude_segments ~w(graph graphics gpu engine render shader texture mesh vertex)
  @exclude_substrings ["opengl", "open_gl"]
  @math3d_tokens ~w(vec2 vec3 vec4 mat2 mat3 mat4 quat quaternion matrix3d transform3d)

  @parallel_categories ~w(kmeans pca neural_network gaussian_nb multinomial_nb)
  @async_categories ~w(
    detect_outliers
    fold
    map
    reduce
    scan
    rolling
    ema
    sma
    wma
    moments
    percentile
    group
    partition
  )

  def all do
    case :persistent_term.get(@entries_cache_key, nil) do
      nil ->
        entries = build_entries()
        :persistent_term.put(@entries_cache_key, entries)
        entries

      entries ->
        entries
    end
  end

  def allowed do
    Enum.filter(all(), & &1.allowed)
  end

  def excluded do
    Enum.reject(all(), & &1.allowed)
  end

  def report do
    case :persistent_term.get(@report_cache_key, nil) do
      nil ->
        entries = all()
        report = build_report(entries)
        :persistent_term.put(@report_cache_key, report)
        report

      report ->
        report
    end
  end

  @doc false
  def clear_cache do
    erase_if_present(@entries_cache_key)
    erase_if_present(@report_cache_key)
    :ok
  end

  defp erase_if_present(key) do
    try do
      :persistent_term.erase(key)
    rescue
      ArgumentError -> :ok
    end
  end

  defp build_entries do
    @source_module.__info__(:functions)
    |> Enum.map(fn {name, arity} ->
      name_str = Atom.to_string(name)
      %{name: name_str, arity: arity}
    end)
    |> Enum.filter(&include_function?/1)
    |> Enum.map(&annotate_entry/1)
    |> Enum.sort_by(& &1.name)
  end

  defp include_function?(%{name: name}) do
    Enum.any?(@include_prefixes, &String.starts_with?(name, &1))
  end

  defp annotate_entry(entry) do
    category = categorize(entry.name)
    mode = mode_for(category)
    reason = exclusion_reason(entry.name)

    entry
    |> Map.put(:category, category)
    |> Map.put(:mode, mode)
    |> Map.put(:allowed, is_nil(reason))
    |> Map.put(:reason, reason)
  end

  defp exclusion_reason(name) do
    segments = String.split(name, "_")

    case match_segment(segments) do
      nil ->
        case Enum.find(@exclude_substrings, &String.contains?(name, &1)) do
          nil -> nil
          token -> "excluded substring: #{token}"
        end

      token ->
        "excluded segment: #{token}"
    end
  end

  defp match_segment(segments) do
    Enum.find_value(@exclude_segments, fn token ->
      if Enum.any?(segments, &segment_matches_token?(&1, token)) do
        token
      else
        nil
      end
    end)
  end

  defp segment_matches_token?(segment, token) do
    segment == token or String.starts_with?(segment, token)
  end

  defp categorize("get_" <> _), do: "accessor"

  defp categorize("fp_" <> rest) do
    parts = String.split(rest, "_")

    cond do
      math3d?(rest, parts) -> "math3d"
      match_parts?(parts, ["detect", "outliers"]) -> "detect_outliers"
      match_parts?(parts, ["rolling"]) -> "rolling"
      match_parts?(parts, ["neural", "network"]) -> "neural_network"
      match_parts?(parts, ["gaussian", "nb"]) -> "gaussian_nb"
      match_parts?(parts, ["multinomial", "nb"]) -> "multinomial_nb"
      match_parts?(parts, ["pca"]) -> "pca"
      match_parts?(parts, ["kmeans"]) -> "kmeans"
      match_parts?(parts, ["percentile"]) -> "percentile"
      match_parts?(parts, ["percentiles"]) -> "percentile"
      match_parts?(parts, ["fold"]) -> "fold"
      match_parts?(parts, ["reduce"]) -> "reduce"
      match_parts?(parts, ["map"]) -> "map"
      match_parts?(parts, ["scan"]) -> "scan"
      match_parts?(parts, ["ema"]) -> "ema"
      match_parts?(parts, ["sma"]) -> "sma"
      match_parts?(parts, ["wma"]) -> "wma"
      match_parts?(parts, ["group"]) -> "group"
      match_parts?(parts, ["partition"]) -> "partition"
      true -> List.first(parts) || "other"
    end
  end

  defp categorize(_), do: "other"

  defp match_parts?(parts, [first | _]) do
    List.first(parts) == first
  end

  defp math3d?(name, parts) do
    Enum.any?(@math3d_tokens, fn token ->
      String.contains?(name, token) or Enum.any?(parts, &(&1 == token))
    end)
  end

  defp mode_for(category) do
    cond do
      category == "accessor" -> "sync"
      category in @parallel_categories -> "parallel"
      category in @async_categories -> "async"
      true -> "sync"
    end
  end

  defp build_report(entries) do
    generated_at = DateTime.utc_now() |> DateTime.truncate(:second)
    report_dir = report_dir()
    file_name = "fp_asm_catalog_#{format_timestamp(generated_at)}.md"
    report_path = Path.join(report_dir, file_name)

    {allowed, excluded} = Enum.split_with(entries, & &1.allowed)
    content = render_report(allowed, excluded, generated_at, report_path)

    report = %{
      generated_at: DateTime.to_iso8601(generated_at),
      report_path: report_path,
      counts: %{
        total: length(entries),
        allowed: length(allowed),
        excluded: length(excluded)
      },
      allowed: allowed,
      excluded: excluded
    }

    case File.mkdir_p(report_dir) do
      :ok ->
        case File.write(report_path, content) do
          :ok -> report
          {:error, reason} -> Map.put(report, :error, inspect(reason))
        end

      {:error, reason} ->
        Map.put(report, :error, inspect(reason))
    end
  end

  defp report_dir do
    case System.get_env("MERKLE_DB_INVENTORY_DIR") do
      nil -> Path.expand(@default_dir, File.cwd!())
      dir -> Path.expand(dir, File.cwd!())
    end
  end

  defp format_timestamp(datetime) do
    datetime
    |> DateTime.to_iso8601()
    |> String.replace("-", "")
    |> String.replace(":", "")
    |> String.replace("T", "_")
    |> String.replace("Z", "Z")
  end

  defp render_report(allowed, excluded, generated_at, report_path) do
    category_lines =
      allowed
      |> category_counts()
      |> Enum.map(fn {category, count} -> "- #{category}: #{count}" end)
      |> Enum.join("\n")

    allowed_lines = format_entries(allowed, false)
    excluded_lines = format_entries(excluded, true)

    """
    # FP_ASM_LIB Catalog (Elixir Bridge)

    Generated (UTC): #{DateTime.to_iso8601(generated_at)}
    Source module: #{inspect(@source_module)}
    Source file: lib/merkle_db/asm.ex
    Report path: #{report_path}
    Total functions: #{length(allowed) + length(excluded)}
    Allowed: #{length(allowed)}
    Excluded: #{length(excluded)}

    ## Exclusion Rules
    - segments: #{Enum.join(@exclude_segments, ", ")}
    - substrings: #{Enum.join(@exclude_substrings, ", ")}

    ## Categories (Allowed)
    #{category_lines}

    ## Allowed Functions
    #{allowed_lines}

    ## Excluded Functions
    #{excluded_lines}
    """
  end

  defp category_counts(functions) do
    functions
    |> Enum.group_by(& &1.category)
    |> Map.new(fn {category, items} -> {category, length(items)} end)
    |> Enum.sort_by(fn {category, _count} -> category end)
    |> Map.new()
  end

  defp format_entries([], _show_reason), do: "- none"

  defp format_entries(entries, show_reason) do
    entries
    |> Enum.map(fn entry -> format_entry(entry, show_reason) end)
    |> Enum.join("\n")
  end

  defp format_entry(entry, show_reason) do
    base = "- #{entry.name}/#{entry.arity} (#{entry.category}, #{entry.mode})"

    if show_reason do
      "#{base} - #{entry.reason}"
    else
      base
    end
  end
end
