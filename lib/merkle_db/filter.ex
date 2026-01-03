defmodule MerkleDb.Filter do
  @moduledoc """
  Payload filtering DSL for MerkleDB.

  Filters allow querying vectors based on their associated payload metadata.

  ## Filter DSL

  Filters are expressed as lists of conditions:

      where: [
        {"category", "==", "electronics"},
        {"price", ">=", 100},
        {"in_stock", "==", true}
      ]

  ## Supported Operators

  - `"=="` - Equal (works with strings, numbers, booleans)
  - `"!="` - Not equal
  - `">"` - Greater than (numbers only)
  - `">="` - Greater than or equal
  - `"<"` - Less than
  - `"<="` - Less than or equal
  - `"in"` - Value in list
  - `"not_in"` - Value not in list
  - `"contains"` - String contains substring
  - `"starts_with"` - String starts with prefix
  - `"exists"` - Field exists (value should be true/false)

  ## Examples

      # Simple equality
      Filter.matches?(%{"type" => "book"}, [{"type", "==", "book"}])
      # => true

      # Numeric comparison
      Filter.matches?(%{"price" => 150}, [{"price", ">=", 100}])
      # => true

      # Multiple conditions (AND)
      Filter.matches?(
        %{"type" => "book", "price" => 25},
        [{"type", "==", "book"}, {"price", "<", 50}]
      )
      # => true
  """

  @type condition :: {String.t(), String.t(), term()}
  @type filter :: [condition()]

  @doc """
  Check if a payload matches all filter conditions.

  Returns `true` if the payload matches all conditions, `false` otherwise.
  Empty filters always match.
  """
  @spec matches?(map(), filter()) :: boolean()
  def matches?(_payload, []), do: true

  def matches?(payload, conditions) when is_map(payload) and is_list(conditions) do
    Enum.all?(conditions, fn condition ->
      match_condition?(payload, condition)
    end)
  end

  def matches?(_, _), do: false

  @doc """
  Parse a filter from a JSON-compatible structure.

  Accepts either:
  - A list of [field, op, value] arrays
  - A map with field => value for equality checks

  ## Examples

      Filter.parse([["price", ">=", 100], ["category", "==", "books"]])
      # => {:ok, [{"price", ">=", 100}, {"category", "==", "books"}]}

      Filter.parse(%{"category" => "books"})
      # => {:ok, [{"category", "==", "books"}]}
  """
  @spec parse(term()) :: {:ok, filter()} | {:error, String.t()}
  def parse(nil), do: {:ok, []}
  def parse([]), do: {:ok, []}

  def parse(conditions) when is_list(conditions) do
    result =
      Enum.reduce_while(conditions, [], fn item, acc ->
        case parse_condition(item) do
          {:ok, condition} -> {:cont, [condition | acc]}
          {:error, reason} -> {:halt, {:error, reason}}
        end
      end)

    case result do
      {:error, reason} -> {:error, reason}
      conditions -> {:ok, Enum.reverse(conditions)}
    end
  end

  def parse(map) when is_map(map) do
    conditions =
      Enum.map(map, fn {field, value} ->
        {to_string(field), "==", value}
      end)

    {:ok, conditions}
  end

  def parse(_), do: {:error, "Invalid filter format"}

  @doc """
  Parse a filter from a URL query string value.

  Accepts JSON-encoded filter string.

  ## Examples

      Filter.parse_query_param(~s|[["price",">=",100]]|)
      # => {:ok, [{"price", ">=", 100}]}
  """
  @spec parse_query_param(String.t() | nil) :: {:ok, filter()} | {:error, String.t()}
  def parse_query_param(nil), do: {:ok, []}
  def parse_query_param(""), do: {:ok, []}

  def parse_query_param(param) when is_binary(param) do
    case Jason.decode(param) do
      {:ok, decoded} -> parse(decoded)
      {:error, _} -> {:error, "Invalid JSON in filter parameter"}
    end
  end

  @doc """
  Validate a filter structure.

  Returns `:ok` if valid, `{:error, reason}` otherwise.
  """
  @spec validate(filter()) :: :ok | {:error, String.t()}
  def validate([]), do: :ok

  def validate(conditions) when is_list(conditions) do
    Enum.reduce_while(conditions, :ok, fn condition, :ok ->
      case validate_condition(condition) do
        :ok -> {:cont, :ok}
        error -> {:halt, error}
      end
    end)
  end

  def validate(_), do: {:error, "Filter must be a list"}

  @doc """
  Create a bitmap mask for records matching the filter.

  Returns a binary where each byte is 1 (match) or 0 (no match).
  This can be used with SIMD operations for fast filtering.
  """
  @spec create_mask([map()], filter()) :: binary()
  def create_mask(payloads, conditions) do
    payloads
    |> Enum.map(fn payload ->
      if matches?(payload, conditions), do: 1, else: 0
    end)
    |> :binary.list_to_bin()
  end

  @doc """
  Apply a filter to a list of {id, payload} tuples.

  Returns the list of IDs that match the filter.
  """
  @spec apply_filter([{term(), map()}], filter()) :: [term()]
  def apply_filter(records, []), do: Enum.map(records, fn {id, _} -> id end)

  def apply_filter(records, conditions) do
    records
    |> Enum.filter(fn {_id, payload} -> matches?(payload, conditions) end)
    |> Enum.map(fn {id, _} -> id end)
  end

  # Private: Match a single condition

  defp match_condition?(payload, {field, op, value}) do
    case get_nested_field(payload, field) do
      nil -> op == "exists" and value == false
      field_value -> compare(field_value, op, value)
    end
  end

  defp match_condition?(_, _), do: false

  # Get a potentially nested field (supports "a.b.c" syntax)
  defp get_nested_field(payload, field) when is_binary(field) do
    parts = String.split(field, ".")
    get_in_path(payload, parts)
  end

  defp get_in_path(value, []), do: value
  defp get_in_path(nil, _), do: nil
  defp get_in_path(map, [key | rest]) when is_map(map) do
    get_in_path(Map.get(map, key), rest)
  end
  defp get_in_path(_, _), do: nil

  # Compare field value to condition value
  defp compare(field_value, "==", value), do: field_value == value
  defp compare(field_value, "!=", value), do: field_value != value

  defp compare(field_value, ">", value) when is_number(field_value) and is_number(value) do
    field_value > value
  end

  defp compare(field_value, ">=", value) when is_number(field_value) and is_number(value) do
    field_value >= value
  end

  defp compare(field_value, "<", value) when is_number(field_value) and is_number(value) do
    field_value < value
  end

  defp compare(field_value, "<=", value) when is_number(field_value) and is_number(value) do
    field_value <= value
  end

  defp compare(field_value, "in", values) when is_list(values) do
    field_value in values
  end

  defp compare(field_value, "not_in", values) when is_list(values) do
    field_value not in values
  end

  defp compare(field_value, "contains", substring)
       when is_binary(field_value) and is_binary(substring) do
    String.contains?(field_value, substring)
  end

  defp compare(field_value, "starts_with", prefix)
       when is_binary(field_value) and is_binary(prefix) do
    String.starts_with?(field_value, prefix)
  end

  defp compare(_field_value, "exists", true), do: true
  defp compare(_field_value, "exists", false), do: false

  defp compare(_, _, _), do: false

  # Parse a single condition
  defp parse_condition([field, op, value]) when is_binary(field) and is_binary(op) do
    {:ok, {field, op, value}}
  end

  defp parse_condition({field, op, value}) when is_binary(field) and is_binary(op) do
    {:ok, {field, op, value}}
  end

  defp parse_condition(item) do
    {:error, "Invalid condition: #{inspect(item)}"}
  end

  # Validate a single condition
  @valid_operators ~w(== != > >= < <= in not_in contains starts_with exists)

  defp validate_condition({field, op, _value})
       when is_binary(field) and op in @valid_operators do
    :ok
  end

  defp validate_condition({field, op, _value}) when is_binary(field) do
    {:error, "Invalid operator: #{inspect(op)}"}
  end

  defp validate_condition(condition) do
    {:error, "Invalid condition format: #{inspect(condition)}"}
  end
end
