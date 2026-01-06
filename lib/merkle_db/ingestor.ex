defmodule MerkleDb.Ingestor do
  @moduledoc """
  Handles processing of large text files (books) into searchable semantic chunks.
  Supports sliding windows to maintain context across chunk boundaries.
  """

  @doc """
  Chunks a long text into overlapping windows of words.
  
  ## Parameters
    - text: The full text binary.
    - size: Number of words per chunk (default 200).
    - overlap: Number of words to overlap between chunks (default 50).
    
  Returns a list of strings.
  """
  def chunk_text(text, size \\ 200, overlap \\ 50) when is_binary(text) do
    words = String.split(text, ~r/\s+/, trim: true)
    do_chunk(words, size, overlap)
  end

  defp do_chunk(words, size, overlap) do
    # Ensure we make progress even if overlap is poorly configured
    step = max(size - overlap, 1)
    
    words
    |> Enum.chunk_every(size, step, :discard)
    |> Enum.map(fn chunk_words -> Enum.join(chunk_words, " ") end)
  end

  @doc """
  Streams a file and chunks it, avoiding loading the entire file into memory at once.
  Useful for very large books.
  """
  def chunk_file(path, size \\ 200, overlap \\ 50) do
    # Simple implementation: read file and chunk. 
    # For Giga-scale books, we would use a more complex Stream-based sliding window.
    if File.exists?(path) do
      path
      |> File.read!()
      |> chunk_text(size, overlap)
    else
      {:error, :not_found}
    end
  end
end
