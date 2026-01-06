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
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :public, {:read_concurrency, true}])
      load_glove()
    else
      :ok
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
    # Note: scaling by 1.0/count
    # NIF signature: fp_map_scale_f32(input_bin, output_size, n, scale)
    ASM.fp_map_scale_f32(sum_vec, @dim * 4, @dim, 1.0 / count)
  end

  defp zero_vector do
    for _ <- 1..@dim, into: <<>>, do: <<0.0::float-little-32>>
  end
end