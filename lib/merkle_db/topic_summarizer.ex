defmodule MerkleDb.TopicSummarizer do
  @moduledoc """
  Service for hierarchical summarization of books.
  Computes 'Topic Vectors' for Passages, Chapters, and the entire Book.
  """

  alias MerkleDb.{Ingestor, TextEmbedding}

  @doc """
  Summarizes a book text into a hierarchical structure of embeddings.
  
  Returns:
  %{ 
    book_vector: binary,
    chapters: [
      %{ 
        title: string,
        chapter_vector: binary,
        passages: [
          %{text: string, passage_vector: binary}
        ]
      }
    ]
  }
  """
  def summarize_book(text, opts \\ []) do
    # 1. Hierarchical chunking
    chapters_data = Ingestor.chunk_book(text, opts)

    # 2. Compute embeddings at all levels
    processed_chapters = Enum.map(chapters_data, fn %{title: title, chunks: chunks} ->
      # Passage level
      passages = Enum.map(chunks, fn chunk_text ->
        %{text: chunk_text, passage_vector: TextEmbedding.embed(chunk_text)}
      end)

      # Chapter level (aggregate passage vectors)
      passage_vectors = Enum.map(passages, & &1.passage_vector)
      chapter_vector = TextEmbedding.summarize_vectors(passage_vectors)

      %{ 
        title: title,
        chapter_vector: chapter_vector,
        passages: passages
      }
    end)

    # 3. Book level (aggregate chapter vectors)
    chapter_vectors = Enum.map(processed_chapters, & &1.chapter_vector)
    book_vector = TextEmbedding.summarize_vectors(chapter_vectors)

    %{ 
      book_vector: book_vector,
      chapters: processed_chapters
    }
  end
end
