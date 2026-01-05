defmodule MerkleDb.SemanticDemo do
  alias MerkleDb.{KV, Query, TextEmbedding, Tree}

  @collection "crime_and_punishment"
  @filename "crime_and_punishment.txt"

  def run do
    IO.puts "=== MerkleDB Semantic Retrieval Demo ==="
    IO.puts "Book: Crime and Punishment"
    
    # 1. Setup
    setup_collection()
    
    # 2. Ingest
    chunks = load_and_chunk()
    IO.puts "Ingesting #{length(chunks)} text chunks..."
    
    {time_ingest, _} = :timer.tc(fn ->
      ingest_chunks(chunks)
    end)
    IO.puts "Ingestion complete in #{Float.round(time_ingest/1000, 2)}ms"
    
    # 3. Build Index
    IO.puts "Building HNSW index..."
    {time_index, _} = :timer.tc(fn ->
      tree = KV.snapshot(@collection)
      # Using slightly higher parameters for better recall on text
      updated_tree = Tree.build_hnsw(tree, m: 32, ef_construction: 128)
      KV.set_tree(@collection, updated_tree)
    end)
    IO.puts "Index built in #{Float.round(time_index/1000, 2)}ms"
    
    # 4. Search Demo
    IO.puts "\n=== Semantic Search Results ==="
    
    queries = [
      "murder of the old woman",
      "Raskolnikov's guilt",
      "Siberia prison",
      "poverty and money",
      "Sonia's faith"
    ]
    
    Enum.each(queries, fn q ->
      search(q)
    end)
  end

  defp setup_collection do
    case KV.create_collection(@collection) do
      :ok -> :ok
      {:error, :already_exists} -> KV.reset(@collection)
    end
  end

  defp load_and_chunk do
    File.read!(@filename)
    |> String.split(["\r\n\r\n", "\n\n"], trim: true) # Split by paragraphs
    |> Enum.map(&String.trim/1)
    |> Enum.filter(fn s -> String.length(s) > 100 end) # Filter short noise
    |> Enum.with_index()
  end

  defp ingest_chunks(chunks) do
    # Batch size of 100 for embedding generation
    chunks
    |> Enum.chunk_every(100)
    |> Enum.each(fn batch ->
      kv_pairs = 
        Enum.map(batch, fn {text, idx} ->
          id = "chap_" <> Integer.to_string(idx)
          vec = TextEmbedding.embed(text)
          # Store text snippet in metadata for retrieval
          meta = %{"snippet" => String.slice(text, 0, 150) <> "..."}
          {id, vec, meta}
        end)
      
      KV.put_batch(@collection, kv_pairs)
      IO.write "."
    end)
    IO.puts ""
  end

  defp search(query_text) do
    IO.puts "\n🔎 Query: \"#{query_text}\""
    query_vec = TextEmbedding.embed(query_text)
    
    {time, results} = :timer.tc(fn ->
      Query.execute(KV.snapshot(@collection), [:knn, query_vec, 5, 0.0])
    end)
    
    IO.puts "   (Found #{length(results)} results in #{Float.round(time/1000, 2)}ms)"
    
    Enum.each(results, fn {key, score} ->
      # Retrieve metadata from tree (conceptually, we'd fetch full text from a KV store)
      # Here we rely on the metadata we stored
      tree = KV.snapshot(@collection)
      
      # Find index for key to lookup metadata
      # (Note: In production, use KV.get or PayloadStore, this is a shortcut)
      idx = Map.get(tree.key_index, key)
      meta = Map.get(tree.metadata, idx, %{})
      snippet = Map.get(meta, "snippet", "No preview")
      
      IO.puts "   - [#{Float.round(score, 4)}] #{key}: \"#{snippet}\""
    end)
  end
end

MerkleDb.SemanticDemo.run()
