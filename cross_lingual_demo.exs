defmodule MerkleDb.CrossLingualDemo do
  alias MerkleDb.{TopicSummarizer, TextEmbedding, ASM}

  def cosine_sim(a_bin, b_bin) do
    ASM.fp_fold_dotp_f32(a_bin, b_bin, 300)
  end

  def run do
    System.put_env("GLOVE_FILE", "data/glove_test.txt")
    text = File.read!("crime_and_punishment.txt")

    IO.puts "=== Cross-Lingual Semantic Search Demo ==="
    IO.puts "Languages: English and Portuguese"
    
    IO.puts "Processing Multilingual Book..."
    summary = TopicSummarizer.summarize_book(text, chunk_size: 10, chunk_overlap: 2)

    # We will search in English
    query_text = "philosophy"
    IO.puts "\n🔎 English Query: \"#{query_text}\""
    query_vec = TextEmbedding.embed(query_text)

    # Find the top relevant passages across all chapters
    IO.puts "\n   --- Top Passages (Any Language) ---"
    
    all_passages = 
      Enum.flat_map(summary.chapters, fn ch ->
        Enum.map(ch.passages, fn p -> {ch.title, p.text, cosine_sim(query_vec, p.passage_vector)} end)
      end)

    all_passages
    |> Enum.sort_by(fn {_, _, sim} -> sim end, :desc)
    |> Enum.take(5)
    |> Enum.each(fn {ch, text, sim} ->
      IO.puts "   - [#{Float.round(sim, 4)}] [#{ch}] \"#{text}\""
    end)

    # Now search in Portuguese for the same concept
    query_text_pt = "filosofia"
    IO.puts "\n🔎 Portuguese Query: \"#{query_text_pt}\""
    query_vec_pt = TextEmbedding.embed(query_text_pt)

    IO.puts "\n   --- Top Passages (Any Language) ---"
    all_passages
    |> Enum.map(fn {ch, text, _} -> {ch, text, cosine_sim(query_vec_pt, TextEmbedding.embed(text))} end)
    |> Enum.sort_by(fn {_, _, sim} -> sim end, :desc)
    |> Enum.take(5)
    |> Enum.each(fn {ch, text, sim} ->
      IO.puts "   - [#{Float.round(sim, 4)}] [#{ch}] \"#{text}\""
    end)
  end
end

MerkleDb.CrossLingualDemo.run()
