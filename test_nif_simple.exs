alias MerkleDb.ASM

IO.puts "Testing ASM.fp_query_gemv_columnar..."

# Create 2 vectors of dimension 4
count = 2
dim = 4

# Columnar layout: tuple of 4 binaries, each with 2 doubles
col0 = <<1.0::little-float-64, 0.0::little-float-64>>
col1 = <<0.0::little-float-64, 1.0::little-float-64>>
col2 = <<1.0::little-float-64, 1.0::little-float-64>>
col3 = <<0.0::little-float-64, 0.0::little-float-64>>

columns = {col0, col1, col2, col3}

# Query vector
query = <<1.0::little-float-64, 0.0::little-float-64, 0.0::little-float-64, 0.0::little-float-64>>

IO.puts "Calling NIF..."
try do
  scores = ASM.fp_query_gemv_columnar(columns, query, count, dim)
  IO.puts "NIF returned."
  IO.inspect(scores, label: "Scores binary")
  
  # Parse scores
  scores_list = for <<s::little-float-64 <- scores>>, do: s
  IO.inspect(scores_list, label: "Scores list")
rescue
  e -> IO.puts "NIF failed: #{inspect(e)}"
end

IO.puts "Testing ASM.fp_query_topk..."
scores_bin = <<0.9::little-float-64, 0.1::little-float-64>>
try do
  {result_count, indices, result_scores} = ASM.fp_query_topk(scores_bin, 2, 1, 0.0)
  IO.puts "TopK returned."
  IO.inspect(result_count, label: "Result count")
  IO.inspect(indices, label: "Indices")
  IO.inspect(result_scores, label: "Scores")
rescue
  e -> IO.puts "TopK failed: #{inspect(e)}"
end
