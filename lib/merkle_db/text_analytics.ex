defmodule MerkleDb.TextAnalytics do
  @moduledoc """
  Advanced text analytics for corpus analysis.
  Provides word frequency, unique words, context extraction, and semantic clustering.
  """
  alias MerkleDb.TextStore

  @doc """
  Analyze the corpus and return comprehensive statistics.

  Returns:
  - total_words: Total word count across all chunks
  - unique_words: Number of distinct words
  - vocabulary_richness: unique/total ratio
  - top_words: Most frequent words with counts
  - avg_chunk_length: Average words per chunk
  """
  def analyze_corpus do
    all_chunks_list = TextStore.get_all()
    all_chunks = Map.new(all_chunks_list)

    if map_size(all_chunks) == 0 do
      {:error, "No text data available. Please ingest first."}
    else
      # Aggregate all text
      all_text =
        all_chunks
        |> Map.values()
        |> Enum.join(" ")

      # Tokenize and count
      words =
        all_text
        |> String.downcase()
        |> String.replace(~r/[^\w\s]/, "")
        |> String.split()

      total_words = length(words)

      # Count word frequencies
      word_freq =
        words
        |> Enum.frequencies()

      unique_words = map_size(word_freq)

      # Top 50 words
      top_words =
        word_freq
        |> Enum.sort_by(fn {_word, count} -> -count end)
        |> Enum.take(50)
        |> Enum.map(fn {word, count} -> %{word: word, count: count} end)

      # Chunk-level stats
      chunk_lengths =
        all_chunks
        |> Map.values()
        |> Enum.map(fn text ->
          text
          |> String.split()
          |> length()
        end)

      avg_chunk_length =
        if length(chunk_lengths) > 0 do
          Float.round(Enum.sum(chunk_lengths) / length(chunk_lengths), 1)
        else
          0.0
        end

      {:ok, %{
        total_words: total_words,
        unique_words: unique_words,
        vocabulary_richness: Float.round(unique_words / max(total_words, 1), 4),
        top_words: top_words,
        avg_chunk_length: avg_chunk_length,
        total_chunks: map_size(all_chunks)
      }}
    end
  end

  @doc """
  Find contexts where a specific word appears.
  Returns chunks containing the word with the word highlighted.
  """
  def find_word_contexts(word, limit \\ 10) do
    word_lower = String.downcase(word)

    TextStore.get_all()
    |> Enum.filter(fn {_key, text} ->
      String.downcase(text) =~ word_lower
    end)
    |> Enum.take(limit)
    |> Enum.map(fn {key, text} ->
      # Highlight the word in context
      highlighted = String.replace(
        text,
        ~r/\b#{Regex.escape(word)}\b/i,
        "**\\0**"
      )

      %{
        chunk_id: key,
        context: highlighted,
        snippet: String.slice(text, 0, 200)
      }
    end)
  end

  @doc """
  Get detailed information about a specific cluster.
  Used for interactive 3D visualization - when user clicks on a point, show cluster details.
  """
  def get_cluster_details(cluster_id) do
    tree = MerkleDb.KV.snapshot()

    if tree.clusters && Map.has_key?(tree.clusters, cluster_id) do
      vec_indices = Map.get(tree.clusters, cluster_id, [])

      # Get all keys and texts from this cluster (up to 10 samples)
      sample_count = min(length(vec_indices), 10)
      sample_indices = Enum.take(vec_indices, sample_count)

      samples =
        sample_indices
        |> Enum.map(fn idx ->
          key = Map.get(tree.keys, idx, "Unknown")
          text = TextStore.get(key) || "(text not available)"

          %{
            chunk_id: key,
            vec_index: idx,
            text: String.slice(text, 0, 300),  # First 300 chars
            full_text: text
          }
        end)

      # Extract common themes from all texts in cluster
      all_texts =
        vec_indices
        |> Enum.take(50)  # Analyze up to 50 for themes
        |> Enum.map(&Map.get(tree.keys, &1))
        |> Enum.filter(&(&1 != nil))
        |> Enum.map(&TextStore.get/1)
        |> Enum.filter(&(&1 != nil))

      themes = extract_common_words(all_texts, 10)

      {:ok, %{
        cluster_id: cluster_id,
        size: length(vec_indices),
        percentage: Float.round(length(vec_indices) / max(tree.count, 1) * 100, 1),
        sample_count: length(samples),
        samples: samples,
        themes: themes,
        theme_summary: Enum.join(themes, ", ")
      }}
    else
      {:error, "Cluster #{cluster_id} not found or database not indexed"}
    end
  end

  @doc """
  Analyze semantic clusters based on topic distribution.
  Returns cluster information if IVF indexing is enabled.
  """
  def analyze_clusters do
    tree = MerkleDb.KV.snapshot()

    if tree.clusters && map_size(tree.clusters) > 0 do
      cluster_stats =
        tree.clusters
        |> Enum.map(fn {cluster_id, vec_indices} ->
          # Get sample texts from this cluster
          sample_keys =
            vec_indices
            |> Enum.take(5)
            |> Enum.map(&Map.get(tree.keys, &1))
            |> Enum.filter(&(&1 != nil))

          sample_texts =
            sample_keys
            |> Enum.map(&TextStore.get/1)
            |> Enum.filter(&(&1 != nil))

          # Extract common words from cluster
          common_words = extract_common_words(sample_texts)

          %{
            cluster_id: cluster_id,
            size: length(vec_indices),
            percentage: Float.round(length(vec_indices) / max(tree.count, 1) * 100, 1),
            sample_texts: sample_texts,
            common_themes: common_words
          }
        end)
        |> Enum.sort_by(fn %{size: size} -> -size end)

      {:ok, %{
        total_clusters: map_size(tree.clusters),
        clusters: cluster_stats,
        indexed: true
      }}
    else
      {:ok, %{
        total_clusters: 0,
        clusters: [],
        indexed: false,
        message: "No IVF clustering available. Database is in flat mode."
      }}
    end
  end

  # Private helpers

  defp extract_common_words(texts, top_n \\ 5) do
    texts
    |> Enum.join(" ")
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split()
    |> Enum.filter(fn word -> String.length(word) > 3 end)  # Filter short words
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_word, count} -> -count end)
    |> Enum.take(top_n)
    |> Enum.map(fn {word, _count} -> word end)
  end
end
