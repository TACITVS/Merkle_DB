defmodule MerkleDb.TopicDemo do
  alias MerkleDb.{TopicSummarizer, TextEmbedding, ASM}

  # Helper for similarity in demo (no DB needed for simple vector comparison)
  def cosine_sim(a_bin, b_bin) do
    # Assuming f32
    # Dot product / (norm_a * norm_b)
    # But our embed/1 already normalizes! So simple dot product is enough.
    # NIF: fp_fold_dotp_f32(a, b, n)
    ASM.fp_fold_dotp_f32(a_bin, b_bin, 300)
  end

  def run do
    System.put_env("GLOVE_FILE", "data/glove_test.txt")
    text = File.read!("crime_and_punishment.txt")

    IO.puts "=== Hierarchical Topic Summarization Demo ==="
    
    IO.puts "Processing Book into Topic Structure..."
    summary = TopicSummarizer.summarize_book(text, chunk_size: 5, chunk_overlap: 2)
    IO.puts "✅ Generated Hierarchy: 1 Book, #{length(summary.chapters)} Chapters."

    query_text = "philosophy"
    IO.puts "\n🔎 Query: \"#{query_text}\""
    query_vec = TextEmbedding.embed(query_text)

    # 1. Book Level Check
    book_sim = cosine_sim(query_vec, summary.book_vector)
    IO.puts "   - Book Similarity: #{Float.round(book_sim, 4)}"

    # 2. Chapter Level Search
    IO.puts "\n   --- Chapter Relevance ---"
    Enum.each(summary.chapters, fn ch ->
      sim = cosine_sim(query_vec, ch.chapter_vector)
      IO.puts "   - [#{Float.round(sim, 4)}] #{ch.title}"
    end)

    # 3. Find best passages in the best chapter
    best_chapter = Enum.max_by(summary.chapters, fn ch -> cosine_sim(query_vec, ch.chapter_vector) end)
    IO.puts "\n   --- Top Passages in #{best_chapter.title} ---"
    
    best_chapter.passages
    |> Enum.map(fn p -> {p.text, cosine_sim(query_vec, p.passage_vector)} end)
    |> Enum.sort_by(fn {_, sim} -> sim end, :desc)
    |> Enum.take(2)
    |> Enum.each(fn {text, sim} ->
      IO.puts "   - [#{Float.round(sim, 4)}] \"#{text}\""
    end)
  end
end

MerkleDb.TopicDemo.run()
