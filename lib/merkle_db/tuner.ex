defmodule MerkleDb.Tuner do
  alias MerkleDb.{KV, Query, TextEmbedding, TextStore}

  # --- PUBLIC API ---

  def run_analysis do
    # 1. Load Corpus
    texts = TextStore.get_all()
    if texts == [], do: throw("Database empty. Ingest first.")

    # 2. Extract Keywords (Frequency Analysis)
    IO.puts("📊 Analyzing Word Frequencies...")
    {common, rare} = extract_keywords(texts)
    test_words = common ++ rare

    # 3. Benchmark Thresholds
    # We test thresholds from 0.15 to 0.45
    results = 
      for threshold <- [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] do
        score = benchmark_threshold(threshold, test_words, texts)
        {threshold, score}
      end

    # 4. Find Winner (Highest threshold that still captures > 85% of keywords)
    best = 
      results
      |> Enum.filter(fn {_, score} -> score >= 0.85 end)
      |> List.last() || {0.20, 1.0} # Default fallback

    %{
      recommended_threshold: elem(best, 0),
      stats: results,
      test_words: test_words
    }
  end

  # --- INTERNAL LOGIC ---

  defp benchmark_threshold(threshold, words, all_texts) do
    # Run test for every word
    scores = 
      Enum.map(words, fn word ->
        # A. CLASSIC SEARCH (The Truth)
        # Find every chunk that actually contains the word
        classic_matches = 
          Enum.filter(all_texts, fn {_, txt} -> 
            String.contains?(String.downcase(txt), String.downcase(word)) 
          end)
          |> Enum.map(fn {k, _} -> k end)
          |> MapSet.new()

        if MapSet.size(classic_matches) == 0 do
          1.0 # Skip words not in text
        else
          # B. VECTOR SEARCH (The Approximation)
          q_vec = TextEmbedding.embed(word)
          root = KV.snapshot()
          # We ask for LOTS of results (1000) to see if the vector engine finds them
          vector_results = Query.execute(root, [:knn, q_vec, 1000, threshold])
          
          vector_matches = 
            vector_results 
            |> Enum.map(fn {k, _, _} -> k end) 
            |> MapSet.new()

          # C. RECALC (Intersection)
          # How many of the "Real" matches did the Vector engine find?
          found = MapSet.intersection(classic_matches, vector_matches)
          MapSet.size(found) / MapSet.size(classic_matches)
        end
      end)
    
    # Return average score for this threshold
    Enum.sum(scores) / length(scores)
  end

  defp extract_keywords(texts) do
    # Simple word counter
    words = 
      texts
      |> Enum.map(fn {_, txt} -> txt end)
      |> Enum.join(" ")
      |> String.downcase()
      |> String.replace(~r/[^\w\s]/, "") # Remove punctuation
      |> String.split()
      |> Enum.frequencies()
      |> Enum.sort_by(fn {_, count} -> count end, :desc)

    # Take Top 5 (Common) and 5 from the middle (Rare-ish)
    common = Enum.take(words, 5) |> Enum.map(&elem(&1, 0))
    
    # Middle of the pack words (not "the", "and", but "temple", "bread")
    mid_index = div(length(words), 5)
    rare = Enum.slice(words, mid_index, 5) |> Enum.map(&elem(&1, 0))

    {common, rare}
  end
end