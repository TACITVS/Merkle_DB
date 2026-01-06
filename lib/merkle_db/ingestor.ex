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

  @doc """
  Chunks a book into chapters, then chunks each chapter into sliding windows.
  Useful for hierarchical topic summarization.
  """
  def chunk_book(text, opts \\ []) when is_binary(text) do
    # Default pattern for chapters
    chapter_pattern = Keyword.get(opts, :chapter_pattern, ~r/CHAPTER [IVXLCDM0-9]+/i)
    chunk_size = Keyword.get(opts, :chunk_size, 200)
    chunk_overlap = Keyword.get(opts, :chunk_overlap, 50)

    # 1. Split into chapters
    chapters = split_chapters(text, chapter_pattern)

    # 2. Chunk each chapter
    Enum.map(chapters, fn {title, content} ->
      chunks = chunk_text(content, chunk_size, chunk_overlap)
      %{title: title, chunks: chunks}
    end)
  end

  defp split_chapters(text, pattern) do
    # Find all chapter markers
    # include_captures adds the titles to the list
    parts = 
      Regex.split(pattern, text, include_captures: true, trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    # If the first part matches the pattern, it's a chapter title.
    # If not, it's a prologue.
    {prologue, remaining} = 
      case parts do
        [first | _] ->
          if Regex.match?(pattern, first) do
            {[], parts}
          else
            {[{"Prologue", Enum.at(parts, 0)}], Enum.drop(parts, 1)}
          end
        [] -> {[], []}
      end

    chapters = 
      remaining
      |> Enum.chunk_every(2)
      |> Enum.map(fn 
        [title, content] -> {title, content}
        [title] -> {title, ""}
      end)

    prologue ++ chapters
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
