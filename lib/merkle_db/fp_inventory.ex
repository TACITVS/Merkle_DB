defmodule MerkleDb.FPInventory do
  @moduledoc false

  @cache_key {__MODULE__, :info}
  @default_dir "docs"
  @source_module MerkleDb.ASM

  def info do
    case :persistent_term.get(@cache_key, nil) do
      nil -> generate()
      cached -> cached
    end
  end

  defp generate do
    functions = inventory_functions()
    generated_at = DateTime.utc_now() |> DateTime.truncate(:second)
    report_dir = inventory_dir()
    file_name = "fp_asm_inventory_#{format_timestamp(generated_at)}.md"
    report_path = Path.join(report_dir, file_name)
    content = render_report(functions, generated_at, report_path)

    info =
      case File.mkdir_p(report_dir) do
        :ok ->
          case File.write(report_path, content) do
            :ok ->
              %{
                count: length(functions),
                generated_at: DateTime.to_iso8601(generated_at),
                report_path: report_path,
                categories: category_counts(functions),
                functions: functions
              }

            {:error, reason} ->
              %{
                count: length(functions),
                generated_at: DateTime.to_iso8601(generated_at),
                report_path: report_path,
                categories: category_counts(functions),
                functions: functions,
                error: inspect(reason)
              }
          end

        {:error, reason} ->
          %{
            count: length(functions),
            generated_at: DateTime.to_iso8601(generated_at),
            report_path: report_path,
            categories: category_counts(functions),
            functions: functions,
            error: inspect(reason)
          }
      end

    :persistent_term.put(@cache_key, info)
    info
  end

  defp inventory_functions do
    @source_module.__info__(:functions)
    |> Enum.map(fn {name, arity} ->
      name_str = Atom.to_string(name)
      %{name: name_str, arity: arity, category: categorize(name_str)}
    end)
    |> Enum.filter(fn %{name: name} ->
      String.starts_with?(name, "fp_") or String.starts_with?(name, "get_")
    end)
    |> Enum.sort_by(& &1.name)
  end

  defp categorize("get_" <> _), do: "accessor"

  defp categorize("fp_" <> rest) do
    parts = String.split(rest, "_")

    case parts do
      ["detect", "outliers" | _] -> "detect_outliers"
      ["rolling" | _] -> "rolling"
      ["neural", "network" | _] -> "neural_network"
      ["gaussian", "nb" | _] -> "gaussian_nb"
      ["multinomial", "nb" | _] -> "multinomial_nb"
      ["pca" | _] -> "pca"
      ["kmeans" | _] -> "kmeans"
      ["fold" | _] -> "fold"
      ["reduce" | _] -> "reduce"
      ["map" | _] -> "map"
      ["scan" | _] -> "scan"
      ["ema" | _] -> "ema"
      ["sma" | _] -> "sma"
      ["wma" | _] -> "wma"
      [first | _] -> first
      _ -> "other"
    end
  end

  defp categorize(_), do: "other"

  defp category_counts(functions) do
    functions
    |> Enum.group_by(& &1.category)
    |> Map.new(fn {category, items} -> {category, length(items)} end)
    |> Enum.sort_by(fn {category, _count} -> category end)
    |> Map.new()
  end

  defp inventory_dir do
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

  defp render_report(functions, generated_at, report_path) do
    categories =
      category_counts(functions)
      |> Enum.map(fn {category, count} -> "- #{category}: #{count}" end)
      |> Enum.join("\n")

    list =
      functions
      |> Enum.map(fn f -> "- #{f.name}/#{f.arity} (#{f.category})" end)
      |> Enum.join("\n")

    """
    # FP_ASM_LIB Inventory (Elixir Bridge)

    Generated (UTC): #{DateTime.to_iso8601(generated_at)}
    Source module: #{inspect(@source_module)}
    Source file: lib/merkle_db/asm.ex
    Report path: #{report_path}
    Total functions: #{length(functions)}

    ## Categories
    #{categories}

    ## Functions
    #{list}
    """
  end
end
