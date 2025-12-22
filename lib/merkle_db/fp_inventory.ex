defmodule MerkleDb.FPInventory do
  @moduledoc false

  alias MerkleDb.FPManifest

  def info do
    report = FPManifest.report()
    functions = Enum.map(report.allowed, &take_function_fields/1)

    info = %{
      count: length(functions),
      generated_at: report.generated_at,
      report_path: report.report_path,
      categories: category_counts(functions),
      functions: functions
    }

    case Map.fetch(report, :error) do
      {:ok, error} -> Map.put(info, :error, error)
      :error -> info
    end
  end

  defp take_function_fields(entry) do
    Map.take(entry, [:name, :arity, :category, :mode])
  end

  defp category_counts(functions) do
    functions
    |> Enum.group_by(& &1.category)
    |> Map.new(fn {category, items} -> {category, length(items)} end)
    |> Enum.sort_by(fn {category, _count} -> category end)
    |> Map.new()
  end
end
