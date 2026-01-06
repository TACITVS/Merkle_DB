defmodule MerkleDb.TextEmbedding do
  @moduledoc """
  High-performance text embedding using GloVe pre-trained vectors.
  Uses ETS for O(1) lookup and AVX2-accelerated NIFs for vector aggregation.
  """

  alias MerkleDb.ASM

  @table :glove_vectors
  @dim 300
  @glove_url "https://nlp.stanford.edu/data/glove.6B.zip"
  
  defp glove_file do
    System.get_env("GLOVE_FILE") || "data/glove.6B.300d.txt"
  end

  # --- API ---

  @doc "Ensures GloVe vectors are loaded into ETS."
  def init do
    case :ets.whereis(@table) do
      :undefined ->
        try do
          :ets.new(@table, [:named_table, :public, {:read_concurrency, true}])
          load_glove()
        rescue
          ArgumentError -> :ok # Already created by another process
        end
      _ -> :ok
    end
  end

  @doc "Loads additional vectors from a file into the existing table."
  def load_vectors(path) do
    init()
    if File.exists?(path) do
      IO.puts "--- Loading Vectors from #{path} into ETS... ---"
      File.stream!(path)
      |> Stream.each(fn line ->
        case String.split(line, " ", trim: true) do
          [word | values] when length(values) == @dim ->
            bin = for v <- values, into: <<>>, do: <<String.to_float(v)::float-little-32>>
            :ets.insert(@table, {word, bin})
          _ -> :ok
        end
      end)
      |> Stream.run()
    else
      {:error, :not_found}
    end
  end

  @doc "Converts a string of text into a fixed-size embedding vector (f32 binary)."
  def embed(text) when is_binary(text) do
    init()

    # Simple tokenizer: lowercase and split by non-alphanumeric
    words = 
      text
      |> String.downcase()
      |> String.split(~r/[^a-z0-9]+/, trim: true)

    # Lookup vectors for each word
    vectors = 
      Enum.reduce(words, [], fn word, acc ->
        case :ets.lookup(@table, word) do
          [{^word, vec}] -> [vec | acc]
          [] -> acc
        end
      end)

    case length(vectors) do
      0 -> zero_vector()
      count -> aggregate_vectors(vectors, count)
    end
  end

  @doc "Aggregates a list of vectors into a single summary vector (f32)."
  def summarize_vectors([]), do: zero_vector()
  def summarize_vectors(vectors) when is_list(vectors) do
    aggregate_vectors(vectors, length(vectors))
  end

  @doc "Converts an f32 binary vector to f64 binary vector."
  def to_f64(vec_f32) do
    for <<f::float-little-32 <- vec_f32>>, into: <<>>, do: <<f::float-little-64>>
  end

  # --- Internal Helpers ---

  defp load_glove do
    file = glove_file()
    if File.exists?(file) do
      IO.puts "--- Loading GloVe Vectors (300d) into ETS... ---"
      File.stream!(file)
      |> Stream.each(fn line ->
        [word | values] = line |> String.trim() |> String.split(" ", trim: true)
        # Convert values to f32 little-endian binary
        bin = for v <- values, into: <<>>, do: <<String.to_float(v)::float-little-32>>
        :ets.insert(@table, {word, bin})
      end)
      |> Stream.run()
      IO.puts "✅ Loaded #{:ets.info(@table, :size)} words into memory."
    else
      IO.puts "❌ GloVe file not found at #{file}"
      IO.puts "Please download it from #{@glove_url} and place the 300d file in data/"
    end
  end

  defp aggregate_vectors(vectors, count) do
    # Concatenate all vectors into one large binary
    # vectors is a list of 300-float binaries (1200 bytes each)
    batch_bin = IO.iodata_to_binary(vectors)

    # Use our AVX2 NIF to sum them all
    sum_vec = ASM.fp_vector_sum_f32(batch_bin, count, @dim)

    # Average by scaling by 1/count
    # Note: Even if we don't scale, we MUST normalize for cosine similarity
    res = 
      if count > 1 do
        ASM.fp_map_scale_f32(sum_vec, @dim * 4, @dim, 1.0 / count)
      else
        sum_vec
      end
    
    l2_normalize(res)
  end

  defp l2_normalize(vec_bin) do
    floats = for <<f::float-little-32 <- vec_bin>>, do: f
    mag = :math.sqrt(Enum.reduce(floats, 0.0, fn x, acc -> acc + x * x end))
    
    if mag > 0 do
      for f <- floats, into: <<>>, do: <<f/mag::float-little-32>>
    else
      vec_bin
    end
  end

  defp zero_vector do
    for _ <- 1..@dim, into: <<>>, do: <<0.0::float-little-32>>
  end
end