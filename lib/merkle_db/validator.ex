defmodule MerkleDb.Validator do
  @moduledoc """
  Input validation for MerkleDb requests.

  Provides validation functions for:
  - Collection names
  - Vector data
  - Query parameters
  - Metadata
  """

  @max_collection_name_length 128
  @max_key_length 1024
  @max_metadata_depth 5
  @max_metadata_keys 100
  @max_string_value_length 10_000
  @max_vector_dimensions 10_000
  @max_batch_size 10_000

  @type validation_result :: :ok | {:error, String.t()}

  # Collection Validation

  @doc """
  Validate a collection name.
  Must be alphanumeric with underscores/hyphens, 1-128 characters.
  """
  @spec validate_collection_name(any()) :: validation_result()
  def validate_collection_name(name) when is_binary(name) do
    cond do
      byte_size(name) == 0 ->
        {:error, "Collection name cannot be empty"}

      byte_size(name) > @max_collection_name_length ->
        {:error, "Collection name too long (max #{@max_collection_name_length} characters)"}

      not Regex.match?(~r/^[a-zA-Z0-9_-]+$/, name) ->
        {:error, "Collection name must contain only alphanumeric characters, underscores, and hyphens"}

      String.contains?(name, "..") ->
        {:error, "Collection name cannot contain path traversal sequences"}

      true ->
        :ok
    end
  end

  def validate_collection_name(_), do: {:error, "Collection name must be a string"}

  # Key Validation

  @doc """
  Validate a vector key/ID.
  """
  @spec validate_key(any()) :: validation_result()
  def validate_key(key) when is_binary(key) do
    cond do
      byte_size(key) == 0 ->
        {:error, "Key cannot be empty"}

      byte_size(key) > @max_key_length ->
        {:error, "Key too long (max #{@max_key_length} characters)"}

      true ->
        :ok
    end
  end

  def validate_key(key) when is_integer(key), do: :ok
  def validate_key(_), do: {:error, "Key must be a string or integer"}

  # Vector Validation

  @doc """
  Validate vector data.
  Must be a list of numbers or a binary.
  """
  @spec validate_vector(any(), keyword()) :: validation_result()
  def validate_vector(vector, opts \\ [])

  def validate_vector(vector, opts) when is_list(vector) do
    expected_dim = Keyword.get(opts, :expected_dim)

    cond do
      length(vector) == 0 ->
        {:error, "Vector cannot be empty"}

      length(vector) > @max_vector_dimensions ->
        {:error, "Vector too large (max #{@max_vector_dimensions} dimensions)"}

      expected_dim != nil and length(vector) != expected_dim ->
        {:error, "Vector dimension mismatch: expected #{expected_dim}, got #{length(vector)}"}

      not Enum.all?(vector, &is_number/1) ->
        {:error, "Vector must contain only numbers"}

      true ->
        :ok
    end
  end

  def validate_vector(vector, opts) when is_binary(vector) do
    expected_dim = Keyword.get(opts, :expected_dim)
    precision = Keyword.get(opts, :precision, :f64)
    elem_size = if precision == :f32, do: 4, else: 8

    actual_dim = div(byte_size(vector), elem_size)

    cond do
      byte_size(vector) == 0 ->
        {:error, "Vector cannot be empty"}

      rem(byte_size(vector), elem_size) != 0 ->
        {:error, "Invalid vector binary size for #{precision} precision"}

      actual_dim > @max_vector_dimensions ->
        {:error, "Vector too large (max #{@max_vector_dimensions} dimensions)"}

      expected_dim != nil and actual_dim != expected_dim ->
        {:error, "Vector dimension mismatch: expected #{expected_dim}, got #{actual_dim}"}

      true ->
        :ok
    end
  end

  def validate_vector(_, _), do: {:error, "Vector must be a list of numbers or a binary"}

  # Query Validation

  @doc """
  Validate query parameters.
  """
  @spec validate_query_params(map()) :: validation_result()
  def validate_query_params(params) when is_map(params) do
    with :ok <- validate_k(params["k"] || params[:k]),
         :ok <- validate_threshold(params["threshold"] || params[:threshold]),
         :ok <- validate_limit(params["limit"] || params[:limit]) do
      :ok
    end
  end

  def validate_query_params(_), do: {:error, "Query params must be a map"}

  defp validate_k(nil), do: :ok
  defp validate_k(k) when is_integer(k) and k > 0 and k <= 10_000, do: :ok
  defp validate_k(k) when is_integer(k) and k <= 0, do: {:error, "k must be positive"}
  defp validate_k(k) when is_integer(k), do: {:error, "k too large (max 10000)"}
  defp validate_k(_), do: {:error, "k must be a positive integer"}

  defp validate_threshold(nil), do: :ok
  defp validate_threshold(t) when is_number(t) and t >= -1.0 and t <= 1.0, do: :ok
  defp validate_threshold(t) when is_number(t), do: {:error, "threshold must be between -1.0 and 1.0"}
  defp validate_threshold(_), do: {:error, "threshold must be a number"}

  defp validate_limit(nil), do: :ok
  defp validate_limit(l) when is_integer(l) and l > 0 and l <= 10_000, do: :ok
  defp validate_limit(l) when is_integer(l) and l <= 0, do: {:error, "limit must be positive"}
  defp validate_limit(l) when is_integer(l), do: {:error, "limit too large (max 10000)"}
  defp validate_limit(_), do: {:error, "limit must be a positive integer"}

  # Metadata Validation

  @doc """
  Validate metadata object.
  Checks depth, key count, and value types.
  """
  @spec validate_metadata(any()) :: validation_result()
  def validate_metadata(nil), do: :ok
  def validate_metadata(meta) when is_map(meta) do
    cond do
      map_size(meta) > @max_metadata_keys ->
        {:error, "Too many metadata keys (max #{@max_metadata_keys})"}

      true ->
        validate_metadata_values(meta, 0)
    end
  end
  def validate_metadata(_), do: {:error, "Metadata must be a map or null"}

  defp validate_metadata_values(_meta, depth) when depth > @max_metadata_depth do
    {:error, "Metadata nested too deeply (max #{@max_metadata_depth} levels)"}
  end

  defp validate_metadata_values(meta, depth) when is_map(meta) do
    Enum.reduce_while(meta, :ok, fn {key, value}, :ok ->
      with :ok <- validate_metadata_key(key),
           :ok <- validate_metadata_value(value, depth + 1) do
        {:cont, :ok}
      else
        error -> {:halt, error}
      end
    end)
  end

  defp validate_metadata_key(key) when is_binary(key) and byte_size(key) <= 256, do: :ok
  defp validate_metadata_key(key) when is_atom(key), do: :ok
  defp validate_metadata_key(key) when is_binary(key), do: {:error, "Metadata key too long: #{key}"}
  defp validate_metadata_key(_), do: {:error, "Metadata keys must be strings"}

  defp validate_metadata_value(value, _depth) when is_number(value), do: :ok
  defp validate_metadata_value(value, _depth) when is_boolean(value), do: :ok
  defp validate_metadata_value(nil, _depth), do: :ok

  defp validate_metadata_value(value, _depth) when is_binary(value) do
    if byte_size(value) <= @max_string_value_length do
      :ok
    else
      {:error, "Metadata string value too long (max #{@max_string_value_length} characters)"}
    end
  end

  defp validate_metadata_value(value, depth) when is_list(value) do
    if length(value) <= 1000 do
      Enum.reduce_while(value, :ok, fn item, :ok ->
        case validate_metadata_value(item, depth) do
          :ok -> {:cont, :ok}
          error -> {:halt, error}
        end
      end)
    else
      {:error, "Metadata array too large (max 1000 items)"}
    end
  end

  defp validate_metadata_value(value, depth) when is_map(value) do
    validate_metadata_values(value, depth)
  end

  defp validate_metadata_value(_, _), do: {:error, "Invalid metadata value type"}

  # Batch Validation

  @doc """
  Validate a batch of vectors.
  """
  @spec validate_batch(any()) :: validation_result()
  def validate_batch(batch) when is_list(batch) do
    cond do
      length(batch) == 0 ->
        {:error, "Batch cannot be empty"}

      length(batch) > @max_batch_size ->
        {:error, "Batch too large (max #{@max_batch_size} vectors)"}

      true ->
        :ok
    end
  end

  def validate_batch(_), do: {:error, "Batch must be a list"}

  # Filter Validation

  @doc """
  Validate query filters.
  """
  @spec validate_filters(any()) :: validation_result()
  def validate_filters(nil), do: :ok
  def validate_filters([]), do: :ok

  def validate_filters(filters) when is_list(filters) do
    valid_ops = [:eq, :neq, :gt, :lt, :gte, :lte, :in, :not_in, :contains, :starts_with, :exists,
                 "==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "exists"]

    Enum.reduce_while(filters, :ok, fn filter, :ok ->
      case filter do
        {field, op, _value} when (is_binary(field) or is_atom(field)) and op in valid_ops ->
          {:cont, :ok}

        [field, op, _value] when (is_binary(field) or is_atom(field)) and op in valid_ops ->
          {:cont, :ok}

        _ ->
          {:halt, {:error, "Invalid filter format: #{inspect(filter)}"}}
      end
    end)
  end

  def validate_filters(_), do: {:error, "Filters must be a list"}

  # Request Body Validation

  @doc """
  Validate request body size.
  """
  @spec validate_body_size(binary(), non_neg_integer()) :: validation_result()
  def validate_body_size(body, max_size) when is_binary(body) do
    if byte_size(body) <= max_size do
      :ok
    else
      {:error, "Request body too large (max #{div(max_size, 1_048_576)}MB)"}
    end
  end

  def validate_body_size(_, _), do: :ok
end
